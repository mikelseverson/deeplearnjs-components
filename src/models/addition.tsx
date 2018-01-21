import { ENV, Scalar, CostReduction, Session, SGDOptimizer, Tensor, FeedEntry, Graph, NDArrayMath, Array1D, InCPUMemoryShuffledInputProviderBuilder } from 'deeplearn';
import * as GraphSerializer from 'deeplearn-graph-serializer'


export class onePlusOneModel {
    session: Session;

    math = ENV.math;
    learningRate = 0.21;
    optimizer: SGDOptimizer;

    // Each training batch will be on this many examples.
    batchSize = 30;

    inputTensor: Tensor;
    targetTensor: Tensor;
    costTensor: Tensor;
    predictionTensor: Tensor;

    graph: Graph

    // Maps tensors to InputProviders.
    feedEntries: FeedEntry[];

    constructor() {
        this.optimizer = new SGDOptimizer(this.learningRate);
        this.graph = this.createAdditionGraph();
        this.session = new Session(this.graph, this.math);
        this.generateTrainingData(10000);
    }

    createAdditionGraph(): Graph {
        const graph = new Graph()
        const inputTensor = graph.placeholder('input addition value', [3]);
        const targetTensor = graph.placeholder('output sum', [1]);
        const predictionTensor = graph.layers.dense(
            `fully_connected_1`,
            inputTensor,
            1,
            (x) => x,
            false
        );
        graph.meanSquaredCost(
            targetTensor,
            predictionTensor
        );
        return graph
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
    train1Batch(shouldFetchCost: boolean): number {
        let costValue = -1;
        this.math.scope(() => {
            const cost = this.session.train(
                this.graph.getNodes()[4].output,
                this.feedEntries,
                this.batchSize,
                this.optimizer,
                shouldFetchCost ? CostReduction.MEAN : CostReduction.NONE
            );
            // Compute the cost, expensive transfer of data
            // from the GPU.
            if (shouldFetchCost) {
                costValue = cost.get();
            }
        });
        return costValue;
    }

    setLearningRate(learningRate: number): void {
        this.learningRate = learningRate;
        if(this.optimizer) {
            this.optimizer.setLearningRate(learningRate)
        }
    }

    setPretrainedModel(): void {
        const json:any = this.getGraphJson()
        json[2].data.values = [1,1,1];
        this.setGraphJson(json);
    }

    getGraphJson(): object {
        return GraphSerializer.graphToJson(this.graph)
    }

    setGraphJson(graphJson: JSON): Graph {
        const graphData = GraphSerializer.jsonToGraph(graphJson) 
        this.graph = graphData.graph
        this.feedEntries[0].tensor = graphData.graph.nodes[0].output
        this.feedEntries[1].tensor = graphData.graph.nodes[1].output
        return this.graph;
    }

    predict(input: number[]): number[] {
        let values;
        this.math.scope((keep, track) => {
            const mapping = [{
                tensor: this.graph.getNodes()[0].output,
                data: Array1D.new(input),
            }];
            values = this.session.eval(this.graph.getNodes()[3].output, mapping).dataSync();
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
                Math.random() - .5,
                Math.random() - .5,
                Math.random() - .5,
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
        const nodes = this.graph.getNodes()
        this.feedEntries = [
            {tensor: nodes[0].output, data: inputProvider},
            {tensor: nodes[1].output, data: targetProvider}
        ];
    }
}