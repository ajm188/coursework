c = [1, 0, 1]';
A = [1, 2, 0;
     1, 0, 2];
b = [5, 6]';
lb = [0, 0, 0];
ub = [];
ctype = "US";
vartype = "CCC";
sense = 1;

[xmin, fmin, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp(xmin);
disp(fmin);
disp(status);
