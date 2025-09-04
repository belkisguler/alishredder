README for AliShredder Data Directory
=====================================

Location: /project/alishredder_data/data_files/

This directory contains data files uploaded and processed by the AliShredder Webinterface (http://alishredder.cibiv.univie.ac.at/). The directory structure is as follows:

- `data_files/`: This subdirectory houses the individual directories. Each directory is uniquely identified by a timestamp and a unique ID.

Example of directory listing:
```
202402261233251009_ee262837-46c6-4777-9126-d08bbed18159
202402261234213382_02b8701e-174d-4fa7-9dba-ee54c8bddd31
202402261346317152_a6c46e11-d092-4313-af1b-ea7a89378920
202402261347185326_82d85c8f-8a15-46cb-afb0-d10797943ad8
```

Cleanup
- Directories within `data_files` that are older than 24 hours are automatically removed by a cron job. (implemented by Robert Happel)
