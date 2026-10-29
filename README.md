# DiskSpaceToFreeFinder

Scans a Mac for disk space that can be freed and writes a Markdown report in the current folder. It does not delete anything.

The scan looks at:

- App, npm, uv, and LM Studio log caches
- The Evernote attachment cache
- Unfinished downloads, including Draw Things `.partial` files
- Repeated LM Studio models, grouped by name after quant and format tags are removed
- Large BiglyBT folders, recordings, backup media, and installer downloads

Photos, Logos, and the rest of an app's real library are listed separately and are not treated as caches.

## Run

```bash
python3 disk_space_report.py
```

That writes `./disk-space-to-free.md`.

```bash
python3 disk_space_report.py -o ~/Desktop/disk-space-to-free.md
python3 disk_space_report.py --self-check
```

The report contains local file paths, so `disk-space-to-free.md` is gitignored.

Requires macOS and Python 3.9 or newer. The standard library is enough.
