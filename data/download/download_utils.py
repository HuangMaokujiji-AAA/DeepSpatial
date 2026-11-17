"""Shared download operations used by the dataset notebooks (no large downloads on import)."""

from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


def project_root(start=None):
    here = Path(start or Path.cwd()).resolve()
    for parent in (here, *here.parents):
        if (parent / "data" / "download").is_dir():
            return parent
    raise FileNotFoundError("请从 DeepSpatial 项目或 data/download 目录运行 notebook。")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _open(url, **headers):
    request = urllib.request.Request(url, headers={"User-Agent": "DeepSpatial-dataset-download/1.0", **headers})
    return urllib.request.urlopen(request, timeout=90)


def _checksum(path, specification):
    algorithm, expected = specification.split(":", 1)
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest().lower() != expected.lower():
        raise ValueError(f"校验值不符：{path}；保留文件供检查，请移走损坏文件后重试。")


def download_file(url, destination, *, size=None, checksum=None):
    """Resume HTTP downloads; promote .part only after length/checksum validation."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + ".download.json")
    if destination.exists():
        previous = json.loads(receipt.read_text()) if receipt.exists() else {}
        if previous and previous["url"] != url:
            raise ValueError(f"目标文件来自另一 URL：{destination}")
        expected_size = size if size is not None else previous.get("size")
        if expected_size is None:
            request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "DeepSpatial-download/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                expected_size = int(response.headers["Content-Length"])
        if destination.stat().st_size != expected_size:
            raise ValueError(f"已存在文件长度不符：{destination}；请移走后重新下载。")
        if checksum:
            _checksum(destination, checksum)
        print(f"已校验，跳过：{destination.name}")
        return destination

    partial = destination.with_name(destination.name + ".part")
    # A complete .part can remain if the kernel was interrupted before rename.
    if size is not None and partial.exists() and partial.stat().st_size == size:
        if checksum:
            _checksum(partial, checksum)
        partial.replace(destination)
        save_json(receipt, {"url": url, "size": size, "checksum": checksum})
        return destination

    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with _open(url, **({"Range": f"bytes={offset}-"} if offset else {})) as response:
                resumed = response.status == 206 and offset > 0
                if resumed:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("服务器返回的续传位置不正确。")
                elif response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith("bytes 0-"):
                        raise ValueError("服务器返回了不完整的起始范围。")
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type and ".html" not in destination.name:
                    raise ValueError(f"下载返回 HTML 登录/错误页面：{url}")
                length = response.headers.get("Content-Length")
                expected = size if size is not None else (int(length) + (offset if resumed else 0) if length else None)
                if size is None and response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    total = content_range.rsplit("/", 1)[-1]
                    if total.isdigit():
                        expected = int(total)
                print(f"下载：{destination.name}，已有 {offset / 1024**2:.1f} MiB", flush=True)
                with partial.open("ab" if resumed else "wb") as stream:
                    progress_time = time.monotonic()
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        stream.write(block)
                        if time.monotonic() - progress_time > 15:
                            print(f"  {stream.tell() / 1024**3:.2f} GiB", flush=True)
                            progress_time = time.monotonic()
            if expected is not None and partial.stat().st_size != expected:
                raise OSError(f"文件未完整下载：{partial.stat().st_size} / {expected}")
            if checksum:
                _checksum(partial, checksum)
            partial.replace(destination)
            save_json(receipt, {"url": url, "size": destination.stat().st_size, "checksum": checksum})
            return destination
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            print(f"连接中断，保留 .part，重试 {attempt + 1}/2：{error}")
            time.sleep(2 ** (attempt + 1))


def _inside(root, name):
    if "\\" in name or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"不安全的归档路径：{name}")
    root = Path(root).resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"归档路径越界：{name}")
    return target


def extract_archive(archive, destination):
    """Extract data files only; reject path traversal and links."""
    archive, destination = Path(archive), Path(destination)
    stamp = destination / ".extraction_complete.json"
    signature = {"archive": str(archive.resolve()), "size": archive.stat().st_size, "mtime_ns": archive.stat().st_mtime_ns}
    if stamp.exists() and json.loads(stamp.read_text()) == signature:
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = _inside(destination, member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"拒绝归档符号链接：{member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as inp, target.open("wb") as out:
                        shutil.copyfileobj(inp, out)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            for member in source:
                target = _inside(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as inp, target.open("wb") as out:
                        shutil.copyfileobj(inp, out)
                else:
                    raise ValueError(f"拒绝归档链接/设备：{member.name}")
    else:
        raise ValueError(f"不是 ZIP/TAR 数据归档：{archive}")
    save_json(stamp, signature)
    return destination


def download_records(records, root, *, extract=False):
    root = Path(root)
    save_json(root / "metadata" / "download_manifest.json", records)
    for item in records:
        target = _inside(root, item["path"])
        download_file(item["url"], target, size=item.get("size"), checksum=item.get("checksum"))
        if extract and target.name.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            extract_archive(target, root / "extracted" / item.get("sample", "archives") / target.name.replace(".", "_"))
    print(f"完成 {len(records)} 个官方文件：{root}")


ALLEN_MANIFEST = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/releases/20241115/manifest.json"


def allen_records(dataset, root, formats=("raw",)):
    root = Path(root)
    manifest_file = download_file(ALLEN_MANIFEST, root / "metadata" / "allen_manifest_20241115.json")
    listing = json.loads(manifest_file.read_text())["file_listing"]
    result = []

    def add(entry, role):
        result.append({"role": role, "url": entry["url"], "path": "official/" + entry["relative_path"],
                       "size": entry["size"], "checksum": "md5:" + entry["file_hash"]})

    matrices = listing[dataset]["expression_matrices"]
    if len(matrices) != 1:
        raise ValueError("预期一个实测表达集合；请检查 manifest。")
    matrix = next(iter(matrices.values()))
    for fmt in formats:
        add(matrix[fmt]["files"]["h5ad"], "expression_" + fmt)
    for key, role in [("gene", "genes"), ("cell_metadata_with_cluster_annotation", "cells")]:
        add(listing[dataset]["metadata"][key]["files"]["csv"], role)
    ccf = listing[dataset + "-CCF"]["metadata"]
    add(ccf["ccf_coordinates"]["files"]["csv"], "ccf")
    if "reconstructed_coordinates" in ccf:
        add(ccf["reconstructed_coordinates"]["files"]["csv"], "reconstructed_coordinates")
    for package, keys in {
        "WMB-taxonomy": ["cluster_to_cluster_annotation_membership_pivoted", "cluster_to_cluster_annotation_membership_color"],
        "Allen-CCF-2020": ["parcellation_to_parcellation_term_membership_acronym", "parcellation_to_parcellation_term_membership_name", "parcellation_term"],
    }.items():
        for key in keys:
            add(listing[package]["metadata"][key]["files"]["csv"], key)
    return result


def zenodo_records(root, record="15240431", filenames=None):
    path = download_file(f"https://zenodo.org/api/records/{record}", Path(root) / "metadata" / f"zenodo_{record}.json")
    metadata = json.loads(path.read_text())
    return [{"url": item["links"]["self"], "path": "official/" + item["key"],
             "size": item["size"], "checksum": item["checksum"]}
            for item in metadata["files"] if filenames is None or item["key"] in filenames]


def parse_geo_soft(text):
    samples, current = [], None
    for line in text.splitlines():
        if line.startswith("^SAMPLE = "):
            current = {"sample": line.split(" = ", 1)[1], "files": []}
            samples.append(current)
        elif current is not None and line.startswith("!Sample_title = "):
            current["title"] = line.split(" = ", 1)[1]
        elif current is not None and re.match(r"!Sample_supplementary_file(?:_\d+)? = ", line):
            url = line.split(" = ", 1)[1]
            if url != "NONE":
                current["files"].append(url.replace("ftp://", "https://", 1))
    return samples


def geo_records(accession, root, sample_ids=None, series_samples=None):
    bucket = accession[:-3] + "nnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/{accession}/soft/{accession}_family.soft.gz"
    path = download_file(url, Path(root) / "metadata" / f"{accession}_family.soft.gz")
    with gzip.open(path, "rt") as stream:
        text = stream.read()
    samples = parse_geo_soft(text)
    save_json(Path(root) / "metadata" / f"{accession}_samples.json", samples)
    if sample_ids is not None:
        absent = set(sample_ids) - {s["sample"] for s in samples}
        if absent:
            raise ValueError(f"GEO 不包含配置的样本：{absent}")
    records = []
    for sample in samples:
        if sample_ids is None or sample["sample"] in sample_ids:
            for url in sample["files"]:
                records.append({"url": url, "path": f"official/{sample['sample']}/" + urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[1]),
                                "sample": sample["sample"], "title": sample.get("title", "")})
    # GEO can place expression bundles at Series level (CRC Xenium does this).
    # *_RAW.tar duplicates individual supplementary files, not an extra raw assay.
    for line in text.splitlines():
        if line.startswith("!Series_supplementary_file = "):
            url = line.split(" = ", 1)[1].replace("ftp://", "https://", 1)
            if url == "NONE" or url.endswith("_RAW.tar"):
                continue
            filename = urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[1])
            sample = next((value for pattern, value in (series_samples or {}).items() if re.search(pattern, filename)), accession)
            if sample_ids is None or sample == accession or sample in sample_ids:
                records.append({"url": url, "path": f"official/{sample}/{filename}", "sample": sample, "title": "Series supplementary"})
    if not records:
        raise ValueError("没有找到 GEO supplementary 文件。")
    return records


HTAN_SNAPSHOT = "https://raw.githubusercontent.com/ncihtan/htan-portal/master/public/syn_data.json.gz"


def htan_records(root, participants=("HT206B1", "HT268B1", "HT397B1"), include_images=True):
    lookup_url = "https://raw.githubusercontent.com/ding-lab/ST_subclone_publication/main/Data_access/Sample_ID_Lookup_table_v1.xlsx"
    download_file(lookup_url, Path(root) / "metadata" / "Sample_ID_Lookup_table_v1.xlsx")
    snapshot = download_file(HTAN_SNAPSHOT, Path(root) / "metadata" / "htan_syn_data.json.gz")
    with gzip.open(snapshot, "rt") as stream:
        atlas = next(a for a in json.load(stream)["atlases"] if a["htan_id"].upper() == "HTA12")
    participant_ids = {"HT206B1": "129", "HT268B1": "140", "HT397B1": "253"}
    unknown = set(participants) - participant_ids.keys()
    if unknown:
        raise ValueError(f"未配置的 WUSTL 样本：{unknown}")
    id_to_name = {participant_ids[name]: name for name in participants}
    output = []
    for name, table in atlas.items():
        allowed = (("Visium" in name and ("Level3" in name or "Level4" in name or "Auxiliary" in name))
                   or ("ScRNA-seq" in name and ("Level3" in name or "Level4" in name))
                   or "Xenium" in name or (include_images and "ImagingLevel" in name))
        if not allowed:
            continue
        for entry in table["record_list"]:
            row = dict(zip(table["column_order"], entry["values"]))
            filename = row.get("Filename") or ""
            if not row.get("entityId"):
                continue
            text = " ".join(str(row.get(k) or "") for k in ["Filename", "HTAN Data File ID", "HTAN Parent Biospecimen ID"])
            ids = set(re.findall(r"HTA12_(\d+)_", text))
            participant = next((p for p in participants if p in filename), None)
            if participant is None:
                participant = next((id_to_name[i] for i in ids if i in id_to_name), None)
            if participant:
                output.append({"synapse_id": row["entityId"], "participant": participant, "component": name.split(":", 1)[-1],
                               "filename": filename, "biospecimen_id": row.get("HTAN Parent Biospecimen ID"),
                               "path": f"official/{participant}/" + filename})
    if not output:
        raise ValueError("HTAN 快照中没有匹配文件；请检查样本 ID 与门户发布版本。")
    save_json(Path(root) / "metadata" / "htan_selected_files.json", output)
    return output


def download_synapse(records, root, token=None):
    import getpass
    import os
    import synapseclient

    token = token or os.environ.get("SYNAPSE_AUTH_TOKEN") or getpass.getpass("Synapse personal access token（仅内存使用，不写入 notebook）: ")
    if not token:
        raise ValueError("需要免费的 Synapse 账户和 personal access token。")
    client = synapseclient.Synapse(cache_root_dir=str(Path(root) / ".synapse_cache"))
    client.login(authToken=token, silent=True)
    failures = []
    for record in records:
        target = _inside(root, record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            entity = client.get(record["synapse_id"], downloadLocation=str(target.parent), ifcollision="overwrite.local")
            received = Path(entity.path)
            if received.resolve() != target.resolve():
                if target.exists():
                    raise FileExistsError(target)
                shutil.move(str(received), target)
            print(record["synapse_id"], target.name)
        except Exception as error:
            failures.append({"synapse_id": record["synapse_id"], "error": str(error)})
    save_json(Path(root) / "metadata" / "download_failures.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} 个 HTAN 文件未完成，详情见 download_failures.json；其余文件已保留。")
