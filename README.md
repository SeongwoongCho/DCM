# DCM
Determinant based on Confusion Matrix

- Introduction

 In the Confusion Matrix, We can know how much certain class is misclassified in the other classes.
 In this document, I consider normalized confusion matrix as an prior conditional probabilistic within classes and use this to determine answer class in the classification problem
 
  - Problem
    
    There's N classes C_1, C_2, ..., C_N
    
    f : trained model(function) with Training Set T 
    
    F : Confusion Matrix with Valid Set V(used while training)
    
    D : normalized(for each row) Confusion Matrix
    
    W : normalized weight of Training Set T
    
    x : fixed input data
    
    Let's assume f(x) = [p_1, p_2, ..., p_N]
    
  - Solution
    
    Before we go on, we can consider D(i,j) as P(argmax[f(x)] = j | y = C_i), and W(i) as P(y = C_i) for prior conditional probability P
    
    I want to maximize P(y = C_k | f(x) = [p_1, p_2, ..., p_N]) for variable k
    
    i.e We use argmax_k P(y = C_k | f(x) = [p_1, p_2, ..., p_N]) as a predict label
    
    Apply bayesian rule on above formula
    
        P(y = C_k | f(x) = [p_1, p_2, ..., p _N]) = sum_j { P(y = C_k, argmax[f(x)] =j | f(x) = [p_1, p_2, ..., p_N])}
    
        = sum_j { P(f(x) = [p_1, p_2, ..., p_N] | y = C_k, argmax[f(x)] = j) * P(argmax[f(x)] = j | y = C_k) * P(y = C_i)
    
        = sum_j { P(f(x) = [p_1, p_2, ..., p_N] | y = C_k, argmax[f(x)] = j) * D(k,j) * W(i)}
        
        = sum_j { S(k,j;l)*D(k,j)*W(i)}
    
    In the term P(f(x) = [p_1, p_2, ..., p_N] | y = C_k, argmax[f(x)] = j),
    
    if j == k, the probablity is proportional to p_k, else the probablity is inversely proportional to p_k
    
    Consider the above property, I set P(f(x) = [p_1, p_2, ..., p_N] | y = C_k, argmax[f(x)] = j) as the following function S(k,j ;l)
    
    (and you can set any other function. please suggest better form)
    
        S(k,j ;l) = p_k if k==j 
    
                  = l(1-p_k) if k!=j
              
    Search l which performs best score on validation set V in the (N+1)-element discrete space [0, 1/N, 2/N, ..., 1] 
    
- Code Usage
