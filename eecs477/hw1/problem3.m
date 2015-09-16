c = [3, -5]';
A = [4, 5;
     6, -6;
     1, 8;];
b = [3, 7, 20]';
lb = [];
ub = [];
ctype = "LSU";
vartype = "II";
sense = -1;

[xmax, fmax, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp(xmax);
