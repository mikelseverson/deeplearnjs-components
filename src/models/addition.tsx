import { ENV, Scalar, CostReduction, Session, SGDOptimizer, Tensor, FeedEntry, Graph, NDArrayMath, Array1D, InCPUMemoryShuffledInputProviderBuilder } from 'deeplearn';


export class onePlusOneModel {
    session: Session;
    
    math = ENV.math;
    learningRate = 0.14;
    optimizer: SGDOptimizer;

    // Each training batch will be on this many examples.
    batchSize = 200;

    inputTensor: Tensor;
    targetTensor: Tensor;
    costTensor: Tensor;
    predictionTensor: Tensor;

    // Maps tensors to InputProviders.
    feedEntries: FeedEntry[];

    constructor() {
        this.optimizer = new SGDOptimizer(this.learningRate);
    }

    /**
     * Constructs the graph of the model. Call this method before training.
     */
    setupSession(): void {
        const graph = new Graph();

        // This tensor contains the input. In this case, it is a scalar.
        this.inputTensor = graph.placeholder('input addition value', [3]);
        // This tensor contains the target.
        this.targetTensor = graph.placeholder('output sum', [1]);

        // Predicted value
        this.predictionTensor = graph.layers.dense(
            `fully_connected_1`,
            this.inputTensor,
            1,
            (x) => x
        );

        // Caculated Error
        this.costTensor = graph.meanSquaredCost(
            this.targetTensor,
            this.predictionTensor
        );

        // Create the session only after constructing the graph.
        this.session = new Session(graph, this.math);

        // Generate the data that will be used to train the model.
        this.generateTrainingData(10000);
    }


    /**
     * Trains one batch for one iteration. Call this method multiple times to
     * progressively train. Calling this function transfers data from the GPU in
     * order to obtain the current loss on training data.
     *
     * If shouldFetchCost is true, returns the mean cost across examples in the
     * batch. Otherwise, returns -1. We should only retrieve the cost now and then
     * because doing so requires transferring data from the GPU.
     */
    train1Batch(shouldFetchCost: boolean, step: number): number {
        this.optimizer.setLearningRate(this.learningRate * Math.pow(0.80, Math.floor(step / 42)));

        // Train 1 batch.
        let costValue = -1;
        this.math.scope(() => {
            const cost = this.session.train(
                this.costTensor,
                this.feedEntries,
                this.batchSize,
                this.optimizer,
                shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE
            );
            // Compute the cost (by calling get), which requires transferring data
            // from the GPU.
            if (shouldFetchCost) {
                costValue = cost.get();
            }
        });
        return costValue;
    }

    predict(input: number[]): number[] {
        let values;
        this.math.scope((keep, track) => {
            const mapping = [{
                tensor: this.inputTensor,
                data: Array1D.new(input),
            }];
            values = this.session.eval(this.predictionTensor, mapping).dataSync();
        });
        return values;
    }

    private createFullyConnectedLayer(graph: Graph, inputLayer: Tensor, layerIndex: number, sizeOfThisLayer: number) {
        return graph.layers.dense(
            `fully_connected_${layerIndex}`,
            inputLayer,
            sizeOfThisLayer,
            (x) => x
        );
    }

    /**
     * Generates data used to train. Creates a feed entry that will later be used
     * to pass data into the model. Generates `exampleCount` data points.
     */
    private generateTrainingData(exampleCount: number) {
        const rawInputs = new Array(exampleCount);
        const targetValues = [];

        for (let i = 0; i < exampleCount; i++) {
            rawInputs[i] = [
                Math.floor(Math.random() * 4),
                Math.floor(Math.random() * 4),
                Math.floor(Math.random() * 4),
            ];
            targetValues[i] = [rawInputs[i][0] + rawInputs[i][1] + rawInputs[i][2]];
        }

        // Store the data within Array1Ds so that learnjs can use it.
        const inputArray: Array1D[] = rawInputs.map(c => Array1D.new(c));
        const targetArray: Array1D[] = targetValues.map(c => Array1D.new(c));

        // This provider will shuffle the training data (and will do so in a way
        // that does not separate the input-target relationship).
        const shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([
            inputArray,
            targetArray
        ]);

        const [inputProvider, targetProvider] =
            shuffledInputProviderBuilder.getInputProviders();

        // Maps tensors to InputProviders.
        this.feedEntries = [
            {tensor: this.inputTensor, data: inputProvider},
            {tensor: this.targetTensor, data: targetProvider}
        ];
    }
}