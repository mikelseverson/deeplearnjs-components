// import {Graph, Session, Tensor, FeedEntry, NDArrayMath, NDArrayMathGPU, SGDOptimizer} from 'deeplearn';


// class model {
//     // Runs training
//     session: Session;

//     // Encapsulates math operations on the CPU and GPU.
//     math: NDArrayMath = new NDArrayMathGPU();

//     // An optimizer with a certain initial learning rate. Used for training.
//     initialLearningRate = 0.042;
//     optimizer: SGDOptimizer;

//     // Each training batch will be on this many examples.
//     batchSize = 300;

//     inputTensor: Tensor;
//     targetTensor: Tensor;
//     costTensor: Tensor;
//     predictionTensor: Tensor;

//     feedEntries: FeedEntry[];

//     constructor() {
//         this.optimizer = new SGDOptimizer(this.initialLearningRate);
//     }

//     setupSession(): void {

//         const graph = new Graph();

//         // This tensor contains the input. In this case, it is a scalar.
//         this.inputTensor = graph.placeholder('input RGB value', [3]);

//         // This tensor contains the target.
//         this.targetTensor = graph.placeholder('output RGB value', [3]);

//         // Create 3 fully connected layers, each with half the number of nodes of
//         // the previous layer. The first one has 64 nodes.
//         let fullyConnectedLayer =
//             this.createFullyConnectedLayer(graph, this.inputTensor, 0, 64);

//         fullyConnectedLayer =
//             this.createFullyConnectedLayer(graph, fullyConnectedLayer, 0, 32);

//         fullyConnectedLayer =
//             this.createFullyConnectedLayer(graph, fullyConnectedLayer, 0, 16);

//         // We will optimize using mean squared loss.
//         this.costTensor =
//             graph.meanSquaredCost(this.targetTensor, this.predictionTensor);

//         // Create the session only after constructing the graph.
//         this.session = new Session(graph, this.math);

//         // Generate the data that will be used to train the model.
//         this.generateTrainingData(1e5);
    
//     }


//     train1Batch(shouldFetchCost: boolean): number {
//         // Every 42 steps, lower the learning rate by 15%.
//         const learningRate =
//             this.initialLearningRate * Math.pow(0.85, Math.floor(step / 42));
//         this.optimizer.setLearningRate(learningRate);
    
//         // Train 1 batch.
//         let costValue = -1;
//         this.math.scope(() => {
//           const cost = this.session.train(
//               this.costTensor, this.feedEntries, this.batchSize, this.optimizer,
//               shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE);
    
//           if (!shouldFetchCost) {
//             // We only train. We do not compute the cost.
//             return;
//           }
    
//           // Compute the cost (by calling get), which requires transferring data
//           // from the GPU.
//           costValue = cost.get();
//         });
//         return costValue;
//       }

//   /**
//    * Generates data used to train. Creates a feed entry that will later be used
//    * to pass data into the model. Generates `exampleCount` data points.
//    */
//   private generateTrainingData(exampleCount: number) {
//     this.math.scope(() => {
//       const rawInputs = new Array(exampleCount);
//       for (let i = 0; i < exampleCount; i++) {
//         rawInputs[i] = [
//           this.generateRandomChannelValue(), this.generateRandomChannelValue(),
//           this.generateRandomChannelValue()
//         ];
//       }

//       // Store the data within Array1Ds so that learnjs can use it.
//       const inputArray: Array1D[] =
//           rawInputs.map(c => Array1D.new(this.normalizeColor(c)));
//       const targetArray: Array1D[] = rawInputs.map(
//           c => Array1D.new(
//               this.normalizeColor(this.computeComplementaryColor(c))));

//       // This provider will shuffle the training data (and will do so in a way
//       // that does not separate the input-target relationship).
//       const shuffledInputProviderBuilder =
//           new InCPUMemoryShuffledInputProviderBuilder(
//               [inputArray, targetArray]);
//       const [inputProvider, targetProvider] =
//           shuffledInputProviderBuilder.getInputProviders();

//       // Maps tensors to InputProviders.
//       this.feedEntries = [
//         {tensor: this.inputTensor, data: inputProvider},
//         {tensor: this.targetTensor, data: targetProvider}
//       ];
//     });
//   }



// }
