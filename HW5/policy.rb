$fsm = {
	:s1 => {  :s1 => {:a => 0.2, :b => 0.9},
		  :s2 => {:a => 0.8, :b => 0.0},
                  :s3 => {:a => 0.0, :b => 0.1}
		},
        :s2 => {  :s1 => {:a => 0.8, :b => 0.0},
		  :s2 => {:a => 0.2, :b => 0.9},
		  :s3 => {:a => 0.0, :b => 0.1}
		},
	:s3 => {
		  :s1 => {:a => 0, :b => 0},
		  :s2 => {:a => 0, :b => 0},
		  :s3 => {:a => 1, :b => 1}
		}
}
$r = {:s1 => -1, :s2 => -2, :s3 => 0}
$states = $fsm.keys
$actions = [:a, :b]

def prob(state, target, action)
  begin
    $fsm[state][target][action] || 0
  rescue
    0
  end
end

def q_value(s, action, gamma, v)
  $states.map{|s_prime| prob(s, s_prime, action)*$r[s_prime] +	gamma * v[s_prime]}.reduce(:+)
end

def policy_iterate(policy,initial_values, gamma)
  v, change = initial_values, false
  until change
    p policy
    change = false
    $states.each do |s|
      v[s] = q_value(s, policy[s], gamma, v)
    end
    $states.each do |s|
      q_best = v[s]
      $actions.each do |action|
        q = q_value(s, action, gamma, v)
        if q > q_best
          policy[s] = action
          q_best = q
          change = true
        end
      end
    end
  end
  return policy
end

p policy_iterate({:s1 => :b, :s2 => :b}, {:s1 => -1, :s2 => -2, :s3 => 0}, 1)
