import numpy as np


Identity = np.array([
    [1,0],
    [0,1]
])
Pauli_X = np.array([
    [0,1],
    [1,0]
])
Pauli_Y = np.array([
    [0,-1j],
    [1j,0]
])
Pauli_Z = np.array([
    [1,0],
    [0,-1]
])
CNOT = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
])








#calculates the expecation value based on initial states, gates, and observable.
def expectationValue (initState, Gate, Observable):
  
    state = Gate @ initState
    return np.conjugate(np.transpose(state)) @ Observable @ state


#Takes indiviual initial states and tensors them together
def joinStates (initStates):
    fullState = 1
    for state in initStates:
        fullState = np.kron(fullState, state)

    return fullState

#right now this only does 1 dimensional input data, can add multi dimensional input in future
def encodingGate(inputData, encodingMatrix): 
    return np.exp((-1j * inputData) * encodingMatrix) 

#puts together the trainable circuit and encoding block into a single matrix.
def fullGate(encodingGate, unitary, layers):
    return np.pow(unitary @ encodingGate, layers)

#repeats tensorproduct n times, used for encoding gate initialization
def repeatedKron(matrix, inputData):
    index = 0
    finalMatrix = 1
    while index < inputData.size:
        finalMatrix = np.kron(finalMatrix, encodingGate(matrix, inputData[index]))
        index += 1
    return finalMatrix / inputData.size



# |00>, 2 qubits initialized in the 0 state
qubitCount = 2
baseState = np.array([1,0])
Gate = CNOT
Observable = np.identity(2 ** qubitCount)
inputData = np.full(shape=(qubitCount,1), fill_value=0) #make whatever input data vector you want, right now just [0,0,..,0]
encoder = Pauli_Z
layers = 1
#Change above lines to adjust everything





#Creates a multi-qubit state from an array of initial single qubit states
initState = joinStates(np.full(shape=(qubitCount, 2), fill_value=baseState))
#Puts together the encoding gate and the unitary gate into a single matrix
finalGate = fullGate(repeatedKron(encoder, inputData), Gate, layers)


print(expectationValue(initState, finalGate, Observable))


