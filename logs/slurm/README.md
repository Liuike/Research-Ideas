# Slurm logs

Oscar jobs write per-task stdout and stderr files to this directory. Log files
are intentionally ignored by Git; this README keeps the directory present in a
fresh clone so Slurm can open its output paths before a batch script starts.
Submit jobs from the project root so these relative scheduler paths resolve here.

Never print credentials or authentication tokens to these logs.
