# PhytoOracle PSII Workflow Instruction Manual

## Getting Started

User **must** provide a argument json file like the `arg_example.json`

## Run the workflow

```
resource_monitor -O log-ps2-makeflow -i 2 -- makeflow -T wq --jx main_workflow.jx --jx-args arg_exmaple.json
```
Or
```
resource_monitor -O log-ps2-makeflow -i 2 -- makeflow -T local --jx main_workflow.jx --jx-args arg_exmaple.json
```

