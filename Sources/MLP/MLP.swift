// The Swift Programming Language
// https://docs.swift.org/swift-book
// The swift-mlx Documentation
// https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx
// Yes I know there is a MLXNN framework, but I want to get familiar with MLX first
import MLX
import MLXRandom
import Cmlx
import Charts
import SwiftUI

public func tensordot(
    _ a: MLXArray, _ b: MLXArray, axes: Int, stream: StreamOrDevice = .default
) -> MLXArray {
    MLXArray(
        mlx_tensordot(
            a.ctx, b.ctx, Int32(axes),
            stream.ctx))
}


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
    
    func forwardPass(
        X: MLXArray,
        weightsInputHidden: MLXArray,
        weightsHiddenOutput: MLXArray)
    -> (output: MLXArray,
        hiddenLayerActivations: MLXArray) {
        
        let hiddenLayerActivations = sigmoid(tensordot(X, weightsInputHidden, axes: 1))
        
        let output = sigmoid(tensordot(hiddenLayerActivations, weightsHiddenOutput, axes: 1))
        
        return (output: output,
                hiddenLayerActivations: hiddenLayerActivations)
    }
    
    func backwardPass(
        X: MLXArray,
        y: MLXArray,
        output: MLXArray,
        weightsHiddenOutput: MLXArray,
        weightsInputHidden: MLXArray,
        hiddenLayerActivation: MLXArray)
    -> (hiddenToOutputWeightGradients: MLXArray,
        inputToHiddenWeightGradients: MLXArray)
    {
        let error = output - y
        let errorToOutputWeightGradients = error * backdropSigmoid(output)
        
        let hiddenToOutputWeightGradients = tensordot(hiddenLayerActivation.T, errorToOutputWeightGradients, axes: 1)
        
        
        let hiddenError = tensordot(errorToOutputWeightGradients, weightsHiddenOutput.T, axes: 1) * backdropSigmoid(hiddenLayerActivation)
        let inputToHiddenWeightGradients = tensordot(X.T, hiddenError, axes: 1)
        
        return (hiddenToOutputWeightGradients: hiddenToOutputWeightGradients,
                inputToHiddenWeightGradients: inputToHiddenWeightGradients)
    }
    
    @MainActor func train(X: MLXArray, y: MLXArray) {
        let inputLayerNeurons = 1
        let hiddenLayerNeurons = 3
        let outputLayerNeurons = 1
        
        var weightsInputHidden = MLXRandom.uniform(0..<1, [inputLayerNeurons, hiddenLayerNeurons])
        var weightsHiddenOutput = MLXRandom.uniform(0..<1, [hiddenLayerNeurons, outputLayerNeurons])
        
        let lr = 0.1
        let epochs = 1000
        
        var losses: [Float32] = []
        
        for ep in 0..<epochs {
            let (output_, hiddenLayerActivation) = forwardPass(X: X, weightsInputHidden: weightsInputHidden, weightsHiddenOutput: weightsHiddenOutput)
            let error = square(output_ - y) / 2
            
            let (hiddenToOutputWeightGradients, inputToHiddenWeightGradients) = backwardPass(
                X: X,
                y: y,
                output: output_,
                weightsHiddenOutput: weightsHiddenOutput,
                weightsInputHidden: weightsInputHidden,
                hiddenLayerActivation: hiddenLayerActivation)
            
            weightsHiddenOutput = weightsHiddenOutput - lr * hiddenToOutputWeightGradients
            weightsInputHidden = weightsInputHidden - lr * inputToHiddenWeightGradients
            
            let epochsLoss = mean(error)
            if ep % 100 == 0 {
                print("Error at epoch \(ep) is \(epochsLoss)")
            }
            let loss: [Float32] = epochsLoss.asArray(Float32.self)
            losses.append(loss[0] as Float)
        }
        
        let (finalPred, _) = forwardPass(X: X, weightsInputHidden: weightsInputHidden, weightsHiddenOutput: weightsHiddenOutput)
        
        print("Weights of the network are: ")
        print("w13, w12, w14", weightsHiddenOutput.T)
        print("w35, w25, w45", weightsInputHidden)
        
        
        let lossChart = Chart(losses.indices, id: \.self) { index in
            LineMark(x: .value("Epochs", index), y: .value("Losses", losses[index]))
        }
        plot(lossChart, name: "LossChart")
        
        let finalPredFloat: [Float] = finalPred.asArray(Float.self)
        let finalPredChart = Chart(finalPredFloat.indices, id: \.self) { index in
            LineMark(x: .value("Index", index), y: .value("Value", finalPredFloat[index])).foregroundStyle(.red)
        }
        plot(finalPredChart, name: "Final Prediction Chart")
        
        let yTrainFloat: [Float] = y.asArray(Float.self)
        print(yTrainFloat)
        let yTrainChart = Chart(yTrainFloat.indices, id: \.self) { index in
            LineMark(x: .value("Train Index", index), y: .value("Train Value", yTrainFloat[index])).foregroundStyle(.blue)
        }
        plot(yTrainChart, name: "YTrain Chart")
    }
    
    @MainActor
    func plot(_ chart: Chart<some ChartContent>, name: String = "chartImage") {
        let renderer = ImageRenderer(content:
            chart
            .chartYAxis {
                AxisMarks(position: .automatic)
            }
            .chartXAxis {
                AxisMarks(position: .automatic)
            }
            .padding(40)
            .frame(width: 500, height: 500)
            .background(Color.white)
        )
        if let chartImage = renderer.nsImage {
            let fileURL = URL(filePath: "./\(name).png") // Replace with your desired path
            if let tiffData = chartImage.tiffRepresentation,
               let bitmap = NSBitmapImageRep(data: tiffData),
               let pngData = bitmap.representation(using: .png, properties: [:]) {
                do {
                    try pngData.write(to: fileURL)
                    print("Chart image saved successfully at \(fileURL.path)")
                } catch {
                    print("Failed to save chart image: \(error)")
                }
            }
        }
    }
    
    
    private init() {}
    
    static func main() {
        let mlp = MLP.shared
        
        let XTrain = MLXArray(converting: [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0, 3.0], [11, 1])
        let yTrain = MLXArray(converting: [0.7312, 0.7339, 0.7438, 0.7832, 0.8903, 0.9820, 0.8114, 0.5937, 0.5219, 0.5049, 0.5002], [11, 1])
        
        mlp.train(X: XTrain, y: yTrain)
    }
}
