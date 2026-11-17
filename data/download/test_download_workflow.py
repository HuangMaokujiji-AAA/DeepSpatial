"""Run with the notebook Python environment: python data/download/test_download_workflow.py."""

import gzip
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

import download_utils as dl
import h5ad_utils as hx


class Response(io.BytesIO):
    def __init__(self, data, status=200, **headers):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data)), **headers}


class DownloadWorkflowCheck(unittest.TestCase):
    def test_resume_and_server_ignoring_range(self):
        payload = b"0123456789"
        with tempfile.TemporaryDirectory() as directory:
            for resumed in (True, False):
                target = Path(directory) / str(resumed)
                target.with_name(target.name + ".part").write_bytes(payload[:4])
                response = Response(payload[4:], 206, **{"Content-Range": "bytes 4-9/10"}) if resumed else Response(payload)
                with patch.object(dl, "_open", return_value=response) as request:
                    dl.download_file("https://example.org/counts.h5", target, size=10,
                                     checksum="md5:" + hashlib.md5(payload).hexdigest())
                    self.assertEqual(request.call_args.kwargs, {"Range": "bytes=4-"})
                self.assertEqual(target.read_bytes(), payload)
                with patch.object(dl, "_open", side_effect=AssertionError("must skip")):
                    dl.download_file("https://example.org/counts.h5", target, size=10)

    def test_incomplete_checksum_and_archive_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "matrix.h5"
            with patch.object(dl, "_open", return_value=Response(b"bad")):
                with self.assertRaises(ValueError):
                    dl.download_file("https://example.org/counts.h5", target, checksum="md5:" + "0" * 32)
            self.assertFalse(target.exists())
            self.assertTrue(target.with_name("matrix.h5.part").exists())
            archive = Path(directory) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("../escape.txt", "not allowed")
            with self.assertRaises(ValueError):
                dl.extract_archive(archive, Path(directory) / "unpacked")
            self.assertFalse((Path(directory) / "escape.txt").exists())

    def test_geo_includes_series_level_xenium(self):
        text = """^SERIES = GSE280314
!Series_supplementary_file = ftp://ftp.ncbi.nlm.nih.gov/GSE280314_RAW.tar
!Series_supplementary_file = ftp://ftp.ncbi.nlm.nih.gov/P1_CRC_outs.zip
^SAMPLE = GSM8594564
!Sample_title = P1 Human Colon Cancer
!Sample_supplementary_file_1 = ftp://ftp.ncbi.nlm.nih.gov/GSM8594564_he.tif.gz
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.soft.gz"
            with gzip.open(source, "wt") as stream:
                stream.write(text)
            with patch.object(dl, "download_file", return_value=source):
                records = dl.geo_records("GSE280314", directory, series_samples={"P1_CRC": "GSM8594564"})
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r["sample"] == "GSM8594564" for r in records))
            self.assertTrue(all(r["url"].startswith("https://") for r in records))
            self.assertFalse(any(r["url"].endswith("_RAW.tar") for r in records))

    def test_allen_aligns_ids_and_retains_missing_ccf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            source = ad.AnnData(np.array([[30, 3], [10, 1], [20, 2], [99, 9]], dtype=np.float32),
                                obs=pd.DataFrame(index=["c", "a", "b", "unannotated"]),
                                var=pd.DataFrame(index=["ens2", "ens1"]))
            source.write_h5ad(root / "raw.h5ad")
            pd.DataFrame({"cell_label": ["b", "a", "c"], "brain_section_label": ["test.002", "test.001", "test.001"],
                          "x": [2., 1., 3.], "y": [20., 10., 30.], "z": [0.2, 0.1, 0.1], "subclass": ["rare", "common", "rare"]}).to_csv(root / "cells.csv", index=False)
            pd.DataFrame({"cell_label": ["a", "c"], "x": [101., 103.], "y": [201., 203.], "z": [301., 303.], "parcellation_index": [1, 2]}).to_csv(root / "ccf.csv", index=False)
            pd.DataFrame({"gene_identifier": ["ens1", "ens2"], "gene_symbol": ["Hdc", "Tbx3"]}).to_csv(root / "genes.csv", index=False)
            records = [{"role": role, "path": name} for role, name in [("expression_raw", "raw.h5ad"), ("cells", "cells.csv"), ("ccf", "ccf.csv"), ("genes", "genes.csv")]]
            paths = hx.export_allen(root, records)
            self.assertEqual(len(paths), 2)
            first = ad.read_h5ad(root / "h5ad/raw/sec_001.h5ad")
            self.assertEqual(first.obs_names.tolist(), ["c", "a"])
            self.assertEqual(first.var_names.tolist(), ["Tbx3", "Hdc"])
            self.assertEqual(first.var["gene_identifier"].tolist(), ["ens2", "ens1"])
            np.testing.assert_array_equal(first.X, [[30, 3], [10, 1]])
            np.testing.assert_array_equal(first.obsm["spatial"], [[3, 30], [1, 10]])
            second = ad.read_h5ad(root / "h5ad/raw/sec_002.h5ad")
            self.assertFalse(second.obs["ccf_matched"].iloc[0])
            self.assertTrue(pd.isna(second.obs["x_ccf"].iloc[0]))
            self.assertEqual(hx.export_allen(root, records), paths)

    @staticmethod
    def tenx(path, barcodes):
        path.parent.mkdir(parents=True, exist_ok=True)
        counts = sparse.csc_matrix([[1, 2], [3, 4], [9, 9]])
        with h5py.File(path, "w") as file:
            matrix = file.create_group("matrix")
            for key, value in [("data", counts.data), ("indices", counts.indices), ("indptr", counts.indptr), ("shape", counts.shape)]:
                matrix[key] = value
            matrix["barcodes"] = np.asarray(barcodes, dtype="S")
            features = matrix.create_group("features")
            features["id"] = np.asarray(["ens1", "ens2", "control"], dtype="S")
            features["name"] = np.asarray(["Hdc", "Tbx3", "control"], dtype="S")
            features["feature_type"] = np.asarray(["Gene Expression", "Gene Expression", "Negative Control Probe"], dtype="S")

    def test_visium_and_xenium_export_use_real_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spot = root / "official/GSM1"
            self.tenx(spot / "sample_filtered_feature_bc_matrix.h5", ["0001", "0002"])
            with h5py.File(spot / "sample_filtered_feature_bc_matrix.h5", "r+") as file:
                file["matrix/features/name"][:] = np.asarray(["Hdc", "Hdc", "control"], dtype="S")
            pd.DataFrame([["0002", 1, 0, 1, 200, 20], ["0001", 1, 0, 0, 100, 10]]).to_csv(spot / "sample_tissue_positions_list.csv", header=False, index=False)
            xenium = root / "extracted/GSM2/outs"
            self.tenx(xenium / "cell_feature_matrix.h5", ["cell-a", "cell-b"])
            pd.DataFrame({"cell_id": ["cell-b", "cell-a"], "x_centroid": [2., 1.], "y_centroid": [20., 10.]}).to_parquet(xenium / "cells.parquet")
            # The same cells metadata in two formats is a valid official bundle.
            pd.read_parquet(xenium / "cells.parquet").to_csv(xenium / "cells.csv.gz", index=False)
            paths = hx.export_spatial_downloads(root)
            self.assertEqual(len(paths), 2)
            spots = ad.read_h5ad(root / "h5ad/GSM1.h5ad")
            self.assertEqual(spots.obs_names.tolist(), ["0001", "0002"])
            self.assertEqual(spots.shape, (2, 2))
            self.assertEqual(spots.var_names.tolist(), ["Hdc", "Hdc-1"])
            self.assertEqual(spots.var["name"].tolist(), ["Hdc", "Hdc"])
            np.testing.assert_array_equal(spots.obsm["spatial"], [[10, 100], [20, 200]])
            cells = ad.read_h5ad(root / "h5ad/GSM2.h5ad")
            np.testing.assert_array_equal(cells.obsm["spatial"], [[1, 10], [2, 20]])
            self.assertNotIn("z_coord_um", cells.obs)
            self.assertEqual(cells.uns["coordinate_units"], "micrometers")

    def test_htan_prefixed_matrix_market(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "official/HT206B1/V10X_level3"
            folder.mkdir(parents=True)
            sample = "HT206B1-U2"
            temporary = folder / "source.mtx"
            mmwrite(temporary, sparse.coo_matrix([[1, 2], [3, 4]]))
            with gzip.open(folder / (sample + "-matrix.mtx.gz"), "wb") as stream:
                stream.write(temporary.read_bytes())
            pd.DataFrame(["spot1", "spot2"]).to_csv(folder / (sample + "-barcodes.tsv.gz"), header=False, index=False, sep="\t")
            pd.DataFrame([["ens1", "Hdc", "Gene Expression"], ["ens2", "Tbx3", "Gene Expression"]]).to_csv(folder / (sample + "-features.tsv.gz"), header=False, index=False, sep="\t")
            pd.DataFrame([["spot1", 1, 0, 0, 100, 10], ["spot2", 1, 0, 1, 200, 20]]).to_csv(folder / (sample + "-tissue_positions_list.csv"), header=False, index=False)
            paths = hx.export_spatial_downloads(root, z_um={sample: 50})
            result = ad.read_h5ad(paths[0])
            np.testing.assert_array_equal(result.X.toarray(), [[1, 3], [2, 4]])
            self.assertEqual(result.obs["z_coord_um"].tolist(), [50, 50])

    def test_xenium_numeric_and_partial_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "official/GSM1"
            official.mkdir(parents=True)
            xenium = root / "extracted/GSM1/outs"
            self.tenx(xenium / "cell_feature_matrix.h5", ["1", "100000"])
            pd.DataFrame({"cell_id": [100000, 1], "x_centroid": [2., 1.],
                          "y_centroid": [20., 10.]}).to_parquet(xenium / "cells.parquet")
            labels = official / "R1_supervised.csv.gz"
            pd.DataFrame({"Barcode": ["1", "1e+05"], "Cluster": ["Tumor", "Stromal"]}).to_csv(labels, index=False)
            pd.DataFrame({"Barcode": ["1", "170429"], "ident": ["unsafe", "other"]}).to_csv(
                official / "R1_unsupervised.csv.gz", index=False)
            outputs = hx.export_spatial_downloads(root)
            result = ad.read_h5ad(outputs[0])
            self.assertEqual(result.obs_names.tolist(), ["1", "100000"])
            self.assertEqual(result.obs["annotation_Cluster"].tolist(), ["Tumor", "Stromal"])
            self.assertNotIn("annotation_ident", result.obs)
            self.assertIn("170429", pd.read_csv(official / "R1_unsupervised.csv.gz", dtype=str)["Barcode"].tolist())
            skipped = json.loads((root / "metadata/h5ad_skipped.json").read_text())
            self.assertEqual(skipped[0]["annotation"], "official/GSM1/R1_unsupervised.csv.gz")

            # A subset with no foreign IDs is safe; unmatched matrix cells stay present.
            outputs[0].unlink()
            pd.DataFrame({"Barcode": ["1e+05"], "Cluster": ["Stromal"]}).to_csv(labels, index=False)
            result = ad.read_h5ad(hx.export_spatial_downloads(root)[0])
            self.assertEqual(result.n_obs, 2)
            self.assertTrue(pd.isna(result.obs.loc["1", "annotation_Cluster"]))
            self.assertEqual(result.obs.loc["100000", "annotation_Cluster"], "Stromal")
            np.testing.assert_array_equal(result.obsm["spatial"], [[1, 10], [2, 20]])

            pd.DataFrame({"Barcode": ["100000", "1e+05"], "Cluster": ["a", "b"]}).to_csv(labels, index=False)
            with self.assertRaisesRegex(ValueError, "重复 cell_id"):
                hx.export_spatial_downloads(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
