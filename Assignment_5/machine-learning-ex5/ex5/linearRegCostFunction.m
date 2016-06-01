function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


J1=0;
J2=0;
J1=sum((X*theta-y).^2);
J1=J1/(2*m);

for i=2:length(theta)
J2=J2+(theta(i)^2);
endfor

J2=(J2*lambda)/(2*m);
J=J1+J2;

for i=1:length(theta)
grad(i)=sum((X*theta-y).*X(:,i));
endfor
grad=grad./m;

for i=2:length(theta)
grad(i)=grad(i)+(lambda*theta(i))/m;
endfor	

% =========================================================================

grad = grad(:);

end
