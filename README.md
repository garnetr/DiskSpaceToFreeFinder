# DiskSpaceToFreeFinder

Scans a Mac for disk space that can be freed and writes a Markdown report in the current folder. It also writes a zsh or bash script, matching the terminal's shell, that shows each section and deletes that section only after you confirm it.

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

That writes `./disk-space-to-free.md` and a delete script for the shell this terminal is running. It walks from the scanner up to Terminal, iTerm, or another terminal app and uses the shell attached to that app. A zsh terminal gets `delete-disk-space.zsh`. A bash terminal gets `delete-disk-space.bash`. `--shell zsh` or `--shell bash` overrides the detection.

The delete script prints one section at a time: the size, the paths, and what deleting that section keeps. Answer `y` or `yes` to delete that section. Press Enter to skip it. Photos, Logos, and the other libraries in the report are not in the script.

```bash
./delete-disk-space.zsh
```

Run whichever filename the scanner wrote. The shebang matches that terminal.

```bash
python3 disk_space_report.py -o ~/Desktop/disk-space-to-free.md --delete-script ~/Desktop/delete-disk-space.zsh --shell zsh
python3 disk_space_report.py --self-check
```

The report and the delete script contain local file paths, so both are gitignored.

Requires macOS and Python 3.9 or newer. The standard library is enough.

## License

[MIT](LICENSE). Copyright (c) 2026 Garnet R. Chaney (https://x.com/IAmTheGarnet1).

The copyright line is part of the license notice, so copies of the project keep that address.
