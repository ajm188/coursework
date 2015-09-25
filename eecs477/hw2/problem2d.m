c = [3, 2, 5]';
A = [5, 3, 1;
     4, 2, 8;
     6, 7, 3;];
b = [-8, 23, 1]';
lb = [-Inf, -Inf, 0];
ub = [4, Inf, Inf];
ctype = "SUL";
vartype = "CCC";
sense = -1;

[xmax, fmax, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp(xmax);
disp(fmax);
disp(status);
