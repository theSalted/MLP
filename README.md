# MLX MLP

A very simple Mult-Layer Perceptron with [MLX](https://github.com/ml-explore/mlx-swift).

Data visualization via Swift Charts and SwiftUI.

see reference results in Reference folder.

- Note: I use this project to get faimilar with MLX, so it doesn't utilize existing implementation of some methods in MLXNN and MLXOptimizer.

## Build Run in CLI

To run in command line, go to the package directory.

Then `chmod` the helper script:

`chmod +x mlx-run.sh`

Then:

`./mlx-run.sh --package MLP`

## Troubleshoot

Try build and run project following CLI instruction.

Alternatively run `xcodebuild build -scheme MLP -destination 'platform=OS X' -derivedDataPath ./.derivedData`

