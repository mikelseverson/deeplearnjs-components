import * as React from "react";
import { onePlusOneModel } from '../models/addition'

export interface AdditionModelState { 
    step: number,
    cost: number,
    input1: number,
    input2: number,
    input3: number,
    prediction: number,
    error: number,
    inferences: Array<Object>
}

export class AdditionModel extends React.Component<{}, AdditionModelState> {
    model: onePlusOneModel
    _frameId: any
    trainingStep: number

    constructor(props: any) {
        super(props)
        this.state = {
            step: 0,
            cost: 0,
            input1: 0,
            input2: 0,
            input3: 0,
            prediction: 0,
            error: 0,
            inferences: [],
        }
        this.trainingStep = 0;
        this.train = this.train.bind(this);
        this.handleInputChange = this.handleInputChange.bind(this);
    }

    componentDidMount() {
        this.model = new onePlusOneModel();
        this.startTraing();
    }
    
    componentWillUnmount() {
        this.stopTraining();
    }

    restartModel() {
        this.trainingStep = 0;
        this.model = new onePlusOneModel();
        this.startTraing();
    }

    startTraing() {
        if( !this._frameId ) {
            this._frameId = window.requestAnimationFrame( this.train );
        }
    }

    train() {
        this.trainingStep++;
        let cost = this.model.train1Batch(this.trainingStep % 5 === 0)
        if(this.trainingStep % 5 === 0) {
            const inference = this.randomInference();
            console.log(inference);
            this.setState(state => ({
                step: this.trainingStep,
                cost,
                inferences: [...state.inferences, inference]
            }))
        }
        // Set up next iteration of the loop
        this._frameId = window.requestAnimationFrame( this.train )
    }

    randomInference() {
        const rawInputs = [
            Math.random() - .5,
            Math.random() - .5,
            Math.random() - .5,
        ];
        const expectedOutput = rawInputs.reduce((sum, value) => sum + value);
        const cost = expectedOutput - this.predict(rawInputs)[0];
        return {
            rawInputs,
            expectedOutput,
            cost
        }
    }

    stopTraining() {
        window.cancelAnimationFrame( this._frameId );
        this._frameId = null;
    }

    predict(input: number[]): number[] {
        return this.model.predict(input)
    }

    setTrainedModel(): void {
        this.model.setPretrainedModel();
    }

    handleInputChange(event: any) {
        this.setState({
            [event.target.name]: event.target.value
        });
    }

    handlePrediction(input: number[]): number {
        let prediction = this.predict(input)[0]
        let expectedOutput = input.reduce((output, val) => output + Number(val), 0)
        this.setState(state => ({
            ...state,
            prediction,
            error: prediction - expectedOutput,
        }));
        return prediction
    }

    render() {
        return <div>
            <h1>Addition Neural Network</h1>
            <h2>Step: {this.state.step}</h2>
            <h2>Cost: {this.state.cost}</h2>
            <button onClick={() => this.stopTraining()}>Stop Training</button>
            <button onClick={() => this.startTraing()}>Start Training</button>
            <button onClick={() => this.setTrainedModel()}>Use Pretrained Model</button>
            <button onClick={() => this.restartModel()}>Restart Training</button>
            <br />
            <input
                name="input1"
                type="number"
                value={this.state.input1}
                onChange={this.handleInputChange}
            />
            <input
                name="input2"
                type="number"
                value={this.state.input2}
                onChange={this.handleInputChange}
            />
            <input
                name="input3"
                type="number"
                value={this.state.input3}
                onChange={this.handleInputChange}
            />
            <button onClick={() => this.handlePrediction([this.state.input1, this.state.input2, this.state.input3])}>
                Predict
            </button>

            {this.state.inferences.map((inference, index) => (
                <p key={index}>
                    <p>
                        {inference.rawInputs.map((input, index) => (
                            <span key={index}>{input}</span>
                        ))}
                    </p>
                    <p>Cost: {inference.cost}</p>
                </p>
            ))}
            <p>{this.state.prediction} - {this.state.error && 'error: ' + this.state.error}</p>
        </div>
    }
}