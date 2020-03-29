# PhytoOracle PSII Workflow Instruction Manual

## Getting Started

User **must** provide `"IRODS_BASE_PATH"`, `"INPUT_DIR"` value in the `main_workflow.jx`

`"IRODS_BASE_PATH"` is the iRODS path to fetch the tarball from

`"INPUT_DIR"` is the directory structure after the untar the tarball and the path that contain all the subdir

## Run the workflow

```
makeflow -T wq --jx main_workflow.jx
```
Or
```
makeflow -T local --jx main_workflow.jx
```

