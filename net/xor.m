tvec = [0.0, 0.0;
       1.0, 0.0; 
       0.0, 1.0; 
       1.0, 1.0]; 

tlab = [0;
        1;
        1;
        0];

Q1={rand(3,2), rand(3,1)}; 
train(Q1,tvec', tlab', tvec', tlab', 10^-3)
