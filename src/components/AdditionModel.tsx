import * as React from "react";
import { onePlusOneModel } from '../models/addition'

export interface AdditionModelState { 
    step: number,
    cost: number,
    input1: number,
    input2: number,
    input3: number,
    prediction: number,
    error: number
}

export class AdditionModel extends React.Component<{}, AdditionModelState> {
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
        }
        this.train = this.train.bind(this);
        this.handleInputChange = this.handleInputChange.bind(this)
    }

    _frameId: any
    model: onePlusOneModel

    componentDidMount() {
        this.model = new onePlusOneModel();
        this.model.setupSession();
        this.startTraing();
    }
    
    componentWillUnmount() {
        this.stopTraining();
    }

    startTraing() {
        if( !this._frameId ) {
            this._frameId = window.requestAnimationFrame( this.train );
        }
    }

    train() {
        this.setState(state => ({
            step: this.state.step + 1,
            cost: this.model.train1Batch(true, 1)
        }))
        // Set up next iteration of the loop
        this._frameId = window.requestAnimationFrame( this.train )
    }
    
    stopTraining() {
        window.cancelAnimationFrame( this._frameId );
        this._frameId = null;
    }

    predict(input: number[]): number[] {
        return this.model.predict(input)
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
            <p>{this.state.prediction} - {this.state.error && 'error: ' + this.state.error}</p>
        </div>
    }
}