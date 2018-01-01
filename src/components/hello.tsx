import * as React from "react";
import { onePlusOneModel } from '../models/addition'

export interface HelloProps { 
    compiler: string;
    framework: string;
}

const model = new onePlusOneModel();
model.setupSession();

export class Hello extends React.Component<HelloProps, {}> {
    constructor(props) {
        super(props);
        this.state = {
            step: 0,
            cost: 0,
            input1: 0,
            input2: 0,
            input3: 0
        }
        this.train = this.train.bind(this);
        this.handleInputChange = this.handleInputChange.bind(this)
    }

    componentDidMount() {
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

    predict(input: number[]): number[] {
        return model.predict(input)
    }

    train() {
        this.setState(state => ({
            step: ++state.step,
            cost: model.train1Batch(1,1)
        }))
        // Set up next iteration of the loop
        this._frameId = window.requestAnimationFrame( this.train )
    }
    
    stopTraining() {
        window.cancelAnimationFrame( this._frameId );
        this._frameId = null;
    }

    handleInputChange(event) {
        const target = event.target;
        const value = target.value;
        const name = target.name;
        this.setState({
            [name]: value
        });
    }

    handlePrediction(input: number[]): number[] {
        let prediction = this.predict(input)[0]
        this.setState(state => ({
            ...state,
            prediction,
            error: prediction - Number(input[0]) - Number(input[1]) - Number(input[2]),
        }));
        return prediction;
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
            <p>{this.state.prediction} - error: {this.state.error}</p>
        </div>
    }
}