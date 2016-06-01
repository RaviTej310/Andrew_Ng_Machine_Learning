function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

n=size(theta);  #THIS ACTUALLY BECOMES n+1 BECAUSE theta is from 0 to n or 1 to n+1
sum1=0;
cost=0;

for i = 1:m
sigmoid(X(i,:)*theta);
sum2=(-1*y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
sum1=sum1+sum2;
end

J=sum1/m;

for i=1:n
sum1=0;
for j=1:m
sum1=sum1+(sigmoid(X(j,:)*theta)-y(j))*X(j,i);
end
grad(i)=sum1/m;
end


% =============================================================

end
