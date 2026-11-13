"""Small atomic files and relocatable content identities."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(folder, value):
    """Resume only identical configurations; never silently overwrite a different run."""
    folder = Path(folder)
    path = folder / 'identity.json'
    if folder.exists() and not path.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Nonempty directory without identity: {folder}')
    if path.exists() and read_json(path) != clean(value):
        raise ValueError(f'Identity changed; choose a new cache/output directory: {folder}')
    write_json(path, value)


def source_identity(project):
    return {str(p.relative_to(project)): digest(p) for base in ('raspatial', 'examples', 'scripts')
            for p in sorted((Path(project) / base).rglob('*.py'))}
