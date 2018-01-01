import * as React from "react";
import * as ReactDOM from "react-dom";
import * as dl from 'deeplearn';
import { AdditionModel } from "./components/AdditionModel";

// console.log(dl)
// const graph = new dl.Graph();
// // const math = new dl.NDArrayMathGPU();
// console.log(graph)

// const yHat = graph.constant(36)
// const y = graph.constant(39)
// const a = graph.subtract(y,yHat)
// const loss = graph.variable('loss', a)
// const sess = new dl.Session(graph, math)


// console.log(sess.eval(a, null))
// console.log(loss);

// const constantInitializer = new dl.ConstantInitializer()

// y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
// y = tf.constant(39, name='y')                    # Define y. Set to 39

// loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

// init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
//                                                  # the loss variable will be initialized and ready to be computed
// with tf.Session() as session:                    # Create a session and print the output
//     session.run(init)                            # Initializes the variables
//     print(session.run(loss))                     # Prints the loss

ReactDOM.render(
    <AdditionModel />,
    document.getElementById("main")
);
