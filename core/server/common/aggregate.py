from functools import reduce
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
    
def aggregate_asynchronous(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    
    weighted_weights = [
        [layer * 0.5 for layer in weights] for weights, _ in results
    ]
    
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]  
    return weights_prime
    
    
def aggregate_asynchronous_alpha(results: List[Tuple[NDArrays, int]], alpha) -> NDArrays:
    weighted_weights = []
    
    counter = 0
    for weights, _ in results:
        #Client Model
        if counter == 0: 
            weighted_weights.append([layer * alpha for layer in weights])
        
        #Server Model
        else: 
            weighted_weights.append([layer * (1.0 - alpha) for layer in weights])
        counter = counter + 1    
    
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]
    
    return weights_prime


def aggregate_median(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute median weight of each layer
    median_w: NDArrays = [
        np.median(np.asarray(layer), axis=0) for layer in zip(*weights)  # type: ignore
    ]
    return median_w


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_qffl(
    parameters: NDArrays, deltas: List[NDArrays], hs_fll: List[NDArrays]
) -> NDArrays:
    """Compute weighted average based on  Q-FFL paper."""
    demominator = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_parameters = [(u - v) * 1.0 for u, v in zip(parameters, updates)]
    return new_parameters
