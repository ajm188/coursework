# Attempt no. 1: 2 regions, 2 contractors, k = 2
c = [1, 100, 20, 30]';
A = [1, 0, 0, 0; # number of experienced teams in r_1
     0, 1, 0, 0; # number of experienced teams in r_2
     1, 0, 1, 0; # number of teams in r_1
     1, 0, 1, 0; # number of teams in r_1
     1, 1, 0, 0; # number of teams from c_1
     0, 0, 1, 1; # number of teams from c_2
     ];
b = [1, # min # of experienced teams in r_1
     1, # min # of experienced teams in r_2
     1,
     1,
     2,
     3,
     ];
lb = [0, 0, 0, 0];
ub = [];
ctype = "LLSSUU";
vartype = "CCCC";
sense = 1;

[xmax, fmax, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp("[x_11, x_12, x_21, x_22]");
disp(xmax');


# Attempt no. 2: 3 regions, 2 contractors, k = 2
c = [1, 100, 20, 30, 15, 10]';
A = [1, 0, 0, 0, 0, 0; # number of experienced teams in r_1
     0, 1, 0, 0, 0, 0; # number of experienced teams in r_2
     0, 0, 1, 0, 0, 0;
     1, 0, 0, 1, 0, 0; # number of teams in r_1
     0, 1, 0, 0, 1, 0; # number of teams in r_2
     0, 0, 1, 0, 0, 1;
     1, 1, 1, 0, 0, 0;
     0, 0, 0, 1, 1, 1;
     ];
b = [1, # min # of experienced teams in r_1
     1, # min # of experienced teams in r_2
     1,
     4,
     1,
     1,
     3,
     3,
     ];
lb = [0, 0, 0, 0, 0, 0];
ub = [];
ctype = "LLLSSSUU";
vartype = "CCCCCC";
sense = 1;

[xmax, fmax, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp("[x_11, x_12, x_13, x_21, x_22, x_23]");
disp(xmax');

# Attempt no. 3: 3 regions, 2 contractors, k = 2
c = [200, 100, 300, 1, 5, 1]';
A = [1, 0, 0, 0, 0, 0; # number of experienced teams in r_1
     0, 1, 0, 0, 0, 0; # number of experienced teams in r_2
     0, 0, 1, 0, 0, 0;
     1, 0, 0, 1, 0, 0; # number of teams in r_1
     0, 1, 0, 0, 1, 0; # number of teams in r_2
     0, 0, 1, 0, 0, 1;
     1, 1, 1, 0, 0, 0;
     0, 0, 0, 1, 1, 1;
     ];
b = [1, # min # of experienced teams in r_1
     1, # min # of experienced teams in r_2
     1,
     4,
     3,
     3,
     3,
     9,
     ];
lb = [0, 0, 0, 0, 0, 0];
ub = [];
ctype = "LLLSSSUU";
vartype = "CCCCCC";
sense = 1;

[xmax, fmax, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp("[x_11, x_12, x_13, x_21, x_22, x_23]");
disp(xmax');

# Attempt no. 4: 3 regions, 2 contractors, k = 2
c = [200, 100, 300, 1, 5, 1]';
A = [1, 0, 0, 0, 0, 0; # number of experienced teams in r_1
     0, 1, 0, 0, 0, 0; # number of experienced teams in r_2
     0, 0, 1, 0, 0, 0;
     1, 0, 0, 1, 0, 0; # number of teams in r_1
     0, 1, 0, 0, 1, 0; # number of teams in r_2
     0, 0, 1, 0, 0, 1;
     1, 1, 1, 0, 0, 0;
     0, 0, 0, 1, 1, 1;
     ];
b = [1, # min # of experienced teams in r_1
     1, # min # of experienced teams in r_2
     1,
     4,
     4,
     4,
     10,
     2,
     ];
lb = [0, 0, 0, 0, 0, 0];
ub = [];
ctype = "LLLSSSUU";
vartype = "CCCCCC";
sense = 1;

[xmax, fmax, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp("[x_11, x_12, x_13, x_21, x_22, x_23]");
disp(xmax');

# Attempt no. 5: 3 regions, 2 contractors, k = 2
c = [1, 5, 1, 100, 500, 100]';
A = [1, 0, 0, 0, 0, 0; # number of experienced teams in r_1
     0, 1, 0, 0, 0, 0; # number of experienced teams in r_2
     0, 0, 1, 0, 0, 0;
     1, 0, 0, 1, 0, 0; # number of teams in r_1
     0, 1, 0, 0, 1, 0; # number of teams in r_2
     0, 0, 1, 0, 0, 1;
     1, 1, 1, 0, 0, 0;
     0, 0, 0, 1, 1, 1;
     ];
b = [1, # min # of experienced teams in r_1
     1, # min # of experienced teams in r_2
     1,
     4,
     4,
     6,
     4,
     11,
     ];
lb = [0, 0, 0, 0, 0, 0];
ub = [];
ctype = "LLLSSSUU";
vartype = "CCCCCC";
sense = 1;

[xmax, fmax, status, extra] = glpk(c, A, b, lb, ub, ctype, vartype, sense);
disp("[x_11, x_12, x_13, x_21, x_22, x_23]");
disp(xmax');
