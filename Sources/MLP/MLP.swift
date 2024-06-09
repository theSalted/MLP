// The Swift Programming Language
// https://docs.swift.org/swift-book
// The swift-mlx Documentation
// https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx
// Yes I know there is a MLXNN framework, but I want to get familiar with MLX first
import MLX
import MLXRandom

/// A dead simple multi-layer perceptron
@main
struct MLP {
    private(set) static var shared = MLP()
    
    func backdropSigmoid(_ x: MLXArray) -> MLXArray {
        return x * (1 - x)
    }
    
    func sigmoid(_ x: MLXArray) -> MLXArray {
        return 1 / (1 + exp(-x))
    }
    
    
    func forwardPass(X: MLXArray, weightsInputHidden: MLXArray, weightsHiddenOutput: MLXArray) -> (MLXArray, MLXArray) {
        return (MLXArray(), MLXArray())
    }
    
    func backwardPass(
        X: MLXArray,
        y: MLXArray,
        output: MLXArray,
        weightsHiddenOutput: MLXArray,
        weightsInputHidden: MLXArray,
        hiddenLayerActivation: MLXArray) -> (MLXArray, MLXArray)
    {
        return (MLXArray(), MLXArray())
    }
    
    func train(X: MLXArray, y: MLXArray) {
        let inputLayerNeurons = 1
        let hiddenLayerNeurons = 3
        let outputLayerNeurons = 1
        
        var weightsInputHidden = MLXRandom.uniform(0..<1, [inputLayerNeurons, hiddenLayerNeurons])
        var weightsHiddenOutput = MLXRandom.uniform(0..<1, [hiddenLayerNeurons, outputLayerNeurons])
        
        let lr = 0.1
        let epochs = 1000
        
        var losses: [MLXArray] = []
        
        for ep in 0..<epochs {
            let (output_, hiddenLayerActivation) = forwardPass(X: X, weightsInputHidden: weightsInputHidden, weightsHiddenOutput: weightsHiddenOutput)
            let error = square(output_ - y) / 2
            
            let (_, errorWrtWeightsInputHidden) = backwardPass(
                X: X,
                y: y,
                output: output_,
                weightsHiddenOutput: weightsHiddenOutput,
                weightsInputHidden: weightsInputHidden,
                hiddenLayerActivation: hiddenLayerActivation)
            
            weightsHiddenOutput = weightsHiddenOutput - lr * errorWrtWeightsInputHidden
            weightsInputHidden = weightsInputHidden - lr * errorWrtWeightsInputHidden
            
            let epochsLoss = mean(error)
            if ep % 100 == 0 {
                print("Error at epoch \(ep) is \(epochsLoss)")
            }
            
            losses.append(epochsLoss)
        }
        
        let (finalPred, _) = forwardPass(X: X, weightsInputHidden: weightsInputHidden, weightsHiddenOutput: weightsHiddenOutput)
        
        print(finalPred)
    }
    
    private init() {}
    
    static func main() {
        let mlp = MLP.shared
        
        let XTrain = MLXArray(converting: [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0, 3.0], [11, 1])
        print(XTrain)
        
        let yTrain = MLXArray(converting: [0.7312, 0.7339, 0.7438, 0.7832, 0.8903, 0.9820, 0.8114, 0.5937, 0.5219, 0.5049, 0.5002], [11, 1])
        print(yTrain)
            
        mlp.train(X: XTrain, y: yTrain)
    }
}
