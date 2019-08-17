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
    
    x : fixed input data
    
    Let's assume f(x) = [p_1, p_2, ..., p_N]
    
  - Solution
     
    I want to maximize P(y = C_k | f(x) = [p_1, p_2, ..., p_N]) for variable k
    
    i.e We use argmax_k P(y = C_k | f(x) = [p_1, p_2, ..., p_N]) as a predict label
    
    
    
- Code Usage
