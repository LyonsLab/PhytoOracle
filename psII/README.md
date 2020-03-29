# PhytoOracle PSII Workflow Instruction Manual

## Getting Started

User **must** provide a argument json file like the `arg_example.json`

## Run the workflow

```
makeflow -T wq --jx main_workflow.jx --jx-args arg_exmaple.json
```
Or
```
makeflow -T local --jx main_workflow.jx --jx-args arg_exmaple.json
```

