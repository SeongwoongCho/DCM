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
    
    W : normalized weight of Valid Set V
    
    x : fixed input data
    
    Let's assume f(x) = [p_1, p_2, ..., p_N]
    
  - Solution
    
    Before we go on, we can consider D(i,j) as P(argmax[f(x)] = j | y = C_i), and W(i) as P(y = C_i) for prior conditional probability P
    
    I want to maximize P(y = C_k | f(x) = [p_1, p_2, ..., p_N]) for variable k
    
    i.e We use argmax_k P(y = C_k | f(x) = [p_1, p_2, ..., p_N]) as a predict label
    
    Apply bayesian rule on above formula
    
        P(y = C_k | f(x) = [p_1, p_2, ..., p _N]) = sum_j { P(y = C_k, argmax[f(x)] =j | f(x) = [p_1, p_2, ..., p_N])}
    
        = sum_j { P(f(x) = [p_1, p_2, ..., p_N] | y = C_k, argmax[f(x)] = j) * P(argmax[f(x)] = j | y = C_k) * P(y = C_k)
    
        = sum_j { P(f(x) = [p_1, p_2, ..., p_N] | y = C_k, argmax[f(x)] = j) * D(k,j) * W(k)}
        
        = sum_j { S(k,j;l)*D(k,j)*W(k)}
    
    In the term P(f(x) = [p_1, p_2, ..., p_N] | y = C_k, argmax[f(x)] = j),
    
    if j == k, the probablity is proportional to p_k, else the probablity is inversely proportional to p_k
    
    Consider the above property, I set P(f(x) = [p_1, p_2, ..., p_N] | y = C_k, argmax[f(x)] = j) as the following function S(k,j ;l)
    
    (and you can set any other function. please suggest better form)
    
        S(k,j ;l) = p_k if k==j 
    
                  = l(1-p_k) if k!=j
              
    Search l which performs best score on validation set V in the (N+1)-element discrete space [0, 1/N, 2/N, ..., 1] 
    
- Usage
      
      from DCM import DCM
      from sklearn.metrics import accuracy_score
      
      weight = [0.3, 0.3, 0.4]
      pred = np.array([[0.1,0.2,0.7],[0.3,0.3,0.4],[0.8,0.1,0.1]])
      true = np.array([2,1,0])
      search_space = [0, 0.33, 0.67, 1]
      
      determinant = DCM(n_classes = 3, weight = weight, predict = pred, true = true, metric = accuracy_score)
      determinant.search(space = search_space)
      
      Test_X = load_test_x() ## some function that load test data
      pred_test =  model(Test_X) ## model is defined on somewhere above
      pred_label = determinant.apply(pred_test)
