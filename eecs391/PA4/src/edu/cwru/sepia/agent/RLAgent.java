package edu.cwru.sepia.agent;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

import java.io.*;
import java.util.*;

public class RLAgent extends Agent {

    /**
     * Set in the constructor. Defines how many learning episodes your agent should run for.
     * When starting an episode. If the count is greater than this value print a message
     * and call sys.exit(0)
     */
    public final int numEpisodes;

    /**
     * List of your footmen and your enemies footmen
     */
    private List<Integer> myFootmen;
    private List<Integer> enemyFootmen;
    
    // maps a footman id to his/her cumulative discounted reward
    private Map<Integer, Double> discountedRewards = new HashMap<Integer, Double>();
    
    private Map<Integer, Integer> currentTargets = new HashMap<Integer, Integer>();
    
    public int currentEpisode = 0;
    public boolean evaluationMode = false;
    public int evaluationCount = 0;

    /**
     * Convenience variable specifying enemy agent number. Use this whenever referring
     * to the enemy agent. We will make sure it is set to the proper number when testing your code.
     */
    public static final int ENEMY_PLAYERNUM = 1;

    /**
     * Set this to whatever size your feature vector is.
     */
    public static final int NUM_FEATURES = 3;

    /** Use this random number generator for your epsilon exploration. When you submit we will
     * change this seed so make sure that your agent works for more than the default seed.
     */
    public final Random random = new Random(12345);

    /**
     * Your Q-function weights.
     */
    public Double[] weights;
    
    public double reward;
    public List<Double> rewards = new ArrayList<Double>();

    /**
     * These variables are set for you according to the assignment definition. You can change them,
     * but it is not recommended. If you do change them please let us know and explain your reasoning for
     * changing them.
     */
    public final double gamma = 0.9;
    public final double learningRate = .0001;
    public final double epsilon = .02;

    public RLAgent(int playernum, String[] args) {
        super(playernum);

        if (args.length >= 1) {
            numEpisodes = Integer.parseInt(args[0]);
            System.out.println("Running " + numEpisodes + " episodes.");
        } else {
            numEpisodes = 10;
            System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
        }

        boolean loadWeights = false;
        if (args.length >= 2) {
            loadWeights = Boolean.parseBoolean(args[1]);
        } else {
            System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
        }

        if (loadWeights) {
            weights = loadWeights();
        } else {
            // initialize weights to random values between -1 and 1
            weights = new Double[NUM_FEATURES];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = random.nextDouble() * 2 - 1;
            }
        }
    }

    /**
     * We've implemented some setup code for your convenience. Change what you need to.
     */
    @Override
    public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {

    	if (!evaluationMode && currentEpisode % 10 == 0) {
    		evaluationMode = true;
    		evaluationCount = 0;
    		reward = 0;
    	}

        // Find all of your units
        myFootmen = new LinkedList<Integer>();
        for (Integer unitId : stateView.getUnitIds(playernum)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                myFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        // Find all of the enemy units
        enemyFootmen = new LinkedList<Integer>();
        for (Integer unitId : stateView.getUnitIds(ENEMY_PLAYERNUM)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                enemyFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        return middleStep(stateView, historyView);
    }

    /**
     * You will need to calculate the reward at each step and update your totals. You will also need to
     * check if an event has occurred. If it has then you will need to update your weights and select a new action.
     *
     * If you are using the footmen vectors you will also need to remove killed units. To do so use the historyView
     * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
     * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
     * turn you should not call this as you will get nothing back.
     *
     * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
     *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
     * }
     *
     * You should also check for completed actions using the history view. Obviously you never want a footman just
     * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
     * have an event whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
     * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
     * you can do something similar to the following. Please be aware that on the first turn you should not call this
     *
     * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
     * for(ActionResult result : actionResults.values()) {
     *     System.out.println(result.toString());
     * }
     *
     * @return New actions to execute or nothing if an event has not occurred.
     */
    @Override
    public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {
        Map<Integer, Action> ourHouse = new HashMap<Integer, Action>(); //in the middle of the street
    	
    	//Calculate the reward
    	for(Integer myFootman : myFootmen){
    		Double cumulativeReward = discountedRewards.get(myFootman);
    		if (cumulativeReward == null) {
    			cumulativeReward = 0.0;
    		}
    		double reward = calculateReward(stateView, historyView, myFootman);
    		if (evaluationMode) {
    			this.reward += reward;
    		}
    		discountedRewards.put(myFootman, cumulativeReward + gamma * reward);
    	}
    	boolean event = false;

    	//Update the total reward
    	if (stateView.getTurnNumber() > 0){
	    	for (DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() - 1)){
	    		if (playernum == deathLog.getController()){
	    			myFootmen.remove(myFootmen.indexOf(deathLog.getDeadUnitID()));
	    			currentTargets.remove(myFootmen);
	    			event = true;
	    		} else if (ENEMY_PLAYERNUM == deathLog.getController()){
	    			enemyFootmen.remove(enemyFootmen.indexOf(deathLog.getDeadUnitID()));
	    			event = true;    			
	    		} 
	    	}
	    	
	    	
	    	Map<Integer,ActionResult> commandFeedback = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
	    	for (ActionResult action : commandFeedback.values()) {
	    		if (action.getFeedback() != ActionFeedback.INCOMPLETE){
	    			event = true;
	    		}
	    	}
    	}
    	
    	if (event && !evaluationMode) {
    		double[] oldWeights = new double[NUM_FEATURES];
    		for (int i = 0; i < NUM_FEATURES; i++) {
    			oldWeights[i] = this.weights[i].doubleValue();
    		}
    		
    		double[] newWeights = new double[NUM_FEATURES];
    		for (Integer footman : myFootmen) {
    			Integer target = currentTargets.get(footman);
    			if (target == null || !enemyFootmen.contains(target)) {
    				Unit.UnitView footmanView = stateView.getUnit(footman);
    				Position myPosition = new Position(footmanView.getXPosition(), footmanView.getYPosition());
    				double closestDistance = Double.POSITIVE_INFINITY;
    				for (Integer enemy : enemyFootmen) {
    					Unit.UnitView enemyView = stateView.getUnit(enemy);
    					Position enemyPosition = new Position(enemyView.getXPosition(), enemyView.getYPosition());
    					double distance = myPosition.chebyshevDistance(enemyPosition);
    					if (distance < closestDistance) {
    						closestDistance = distance;
    						target = enemy;
    					}
    				}
    			}
    			double[] oldFeatures = calculateFeatureVector(stateView, historyView, footman, target);
    			newWeights = updateWeights(oldWeights, oldFeatures, discountedRewards.get(footman), stateView, historyView, footman);
    			oldWeights = newWeights;
    		}
    		
    		for (int i = 0; i < NUM_FEATURES; i++) {
    			this.weights[i] = newWeights[i];
    		}
    	}
    	
    	if (event || stateView.getTurnNumber() == 0){
    		for(Integer footman : myFootmen){
    			ourHouse.put(footman, Action.createCompoundAttack(footman, selectAction(stateView, historyView, footman)));
    			
    		}
    	}
    	
    	return ourHouse; //get out of our house
    }

    /**
     * Here you will calculate the cumulative average rewards for your testing episodes. If you have just
     * finished a set of test episodes you will call out testEpisode.
     *
     * It is also a good idea to save your weights with the saveWeights function.
     */
    @Override
    public void terminalStep(State.StateView stateView, History.HistoryView historyView) {
    	if (currentEpisode > numEpisodes) {
    		System.out.println("All done!");
    		System.exit(0);
    	}
    	
        // TODO: MAKE SURE YOU CALL printTestData after you finish a test episode.
    	if (evaluationMode) {
    		evaluationCount++;
    		if (evaluationCount == 5) {
    			evaluationMode = false;
    			evaluationCount = 0;
    			currentEpisode++;
    			rewards.add(reward/5);
    			printTestData(rewards);
    		}
    	} else {
    		currentEpisode++;
    	}

        // Save your weights
        saveWeights(weights);

    }

    /**
     * Calculate the updated weights for this agent. 
     * @param oldWeights Weights prior to update
     * @param oldFeatures Features from (s,a)
     * @param totalReward Cumulative discounted reward for this footman.
     * @param stateView Current state of the game.
     * @param historyView History of the game up until this point
     * @param footmanId The footman we are updating the weights for
     * @return The updated weight vector.
     */
    public double[] updateWeights(double[] oldWeights, double[] oldFeatures, double totalReward, State.StateView stateView, History.HistoryView historyView, int footmanId) {
    	double[] newWeights = new double[oldWeights.length];
    	
    	double reward = calculateReward(stateView, historyView, footmanId);
    	double bestQ = Double.NEGATIVE_INFINITY;
    	for (Integer enemy : enemyFootmen) {
    		double q = calcQValue(stateView, historyView, footmanId, enemy);
    		if (q > bestQ) {
    			bestQ = q;
    		}
    	}
    	for (int i = 0; i < oldWeights.length; i++) {
    		newWeights[i] = oldWeights[i] + learningRate * ((reward + gamma * totalReward) - bestQ) * oldFeatures[i];
    	}

        return newWeights;
    }

    /**
     * Given a footman and the current state and history of the game select the enemy that this unit should
     * attack. This is where you would do the epsilon-greedy action selection.
     *
     * @param stateView Current state of the game
     * @param historyView The entire history of this episode
     * @param attackerId The footman that will be attacking
     * @return The enemy footman ID this unit should attack
     */
    public int selectAction(State.StateView stateView, History.HistoryView historyView, int attackerId) {
    	double randomDouble = this.random.nextDouble();
    	if (randomDouble < this.epsilon) {
    		// exploration: pick a random enemy
    		return this.random.nextInt(this.enemyFootmen.size());
    	} else {
    		int best = -1;
    		double bestValue = Double.NEGATIVE_INFINITY;
    		for (Integer enemyFootmanID : enemyFootmen) {
    			double qValue = calcQValue(stateView, historyView, attackerId, enemyFootmanID);
    			if (qValue > bestValue) {
    				best = enemyFootmanID;
    				bestValue = qValue;
    			}
    		}
    		return best;
    	}
    }

    /**
     * Given the current state and the footman in question calculate the reward received on the last turn.
     * This is where you will check for things like Did this footman take or give damage? Did this footman die
     * or kill its enemy. Did this footman start an action on the last turn? See the assignment description
     * for the full list of rewards.
     *
     * Remember that you will need to discount this reward based on the timestep it is received on. See
     * the assignment description for more details.
     *
     * As part of the reward you will need to calculate if any of the units have taken damage. You can use
     * the history view to get a list of damages dealt in the previous turn. Use something like the following.
     *
     * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
     *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
     *     damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
     *     "attacking unit: " + damageLog.getAttackerID());
     * }
     *
     * You will do something similar for the deaths. See the middle step documentation for a snippet
     * showing how to use the deathLogs.
     *
     * To see if a command was issued you can check the commands issued log.
     *
     * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
     * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
     *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
     * }
     *
     * @param stateView The current state of the game.
     * @param historyView History of the episode up until this turn.
     * @param footmanId The footman ID you are looking for the reward from.
     * @return The current reward
     */
    public double calculateReward(State.StateView stateView, History.HistoryView historyView, int footmanId) {
        double reward = -0.1; // -0.1 for each "step". specified in the assignment description
        
        Set<Integer> unitsIAttacked = new HashSet<Integer>();
        for (DamageLog damageLog : historyView.getDamageLogs(stateView.getTurnNumber() - 1)) {
        	if (damageLog.getAttackerID() == footmanId) {
        		reward += damageLog.getDamage();
        		unitsIAttacked.add(damageLog.getDefenderID());
        	}
        	
        	if (damageLog.getDefenderID() == footmanId) {
        		reward -= damageLog.getDamage();
        	}
        }
        
        for (DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() - 1)) {
        	if (unitsIAttacked.contains(deathLog.getDeadUnitID())) {
        		reward += 100;
        	}
        }
        
        return reward;
    }

    /**
     * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
     * state view and the history of this episode. The action is the attacker and the enemy pair for the
     * SEPIA attack action.
     *
     * This returns the Q-value according to your feature approximation. This is where you will calculate
     * your features and multiply them by your current weights to get the approximate Q-value.
     *
     * @param stateView Current SEPIA state
     * @param historyView Episode history up to this point in the game
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman that your footman would be attacking
     * @return The approximate Q-value
     */
    public double calcQValue(State.StateView stateView,
                             History.HistoryView historyView,
                             int attackerId,
                             int defenderId) {
        double qValue = 0.0;
        double[] featureVector = calculateFeatureVector(stateView, historyView, attackerId, defenderId);
        for (int i = 0; i < this.weights.length; i++) {
        	qValue += this.weights[i] * featureVector[i];
        }
        
        return qValue;
    }

    /**
     * Given a state and action calculate your features here. Please include a comment explaining what features
     * you chose and why you chose them.
     *
     * All of your feature functions should evaluate to a double. Collect all of these into an array. You will
     * take a dot product of this array with the weights array to get a Q-value for a given state action.
     *
     * It is a good idea to make the first value in your array a constant. This just helps remove any offset
     * from 0 in the Q-function. The other features are up to you. Many are suggested in the assignment
     * description.
     *
     * @param stateView Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman. The one you are considering attacking.
     * @return The array of feature function outputs.
     */
    public double[] calculateFeatureVector(State.StateView stateView,
                                           History.HistoryView historyView,
                                           int attackerId,
                                           int defenderId) {
        /* Features: (F attacking E)
    	*	0. # of other units attacking E
    	*	1. 1 if E attacking F (0 otherwise)
    	*	2. 1/chebychev 
    	*/
    	double[] featureVectors = new double[NUM_FEATURES];
    	Unit.UnitView defender = stateView.getUnit(defenderId);
    	Position defenderPos = new Position(defender.getXPosition(), defender.getYPosition());
    	for(Integer footmanID : myFootmen)
    	{
    		Unit.UnitView ally = stateView.getUnit(footmanID);
    		Position p = new Position(ally.getXPosition(), ally.getYPosition());
    		if (attackerId == footmanID){
				if (p.isAdjacent(defenderPos)){
					featureVectors[0] += 1.0;
					featureVectors[1] = 1.0;	
				}
	    		featureVectors[2] = 1/p.chebyshevDistance(defenderPos);
    		} else {
    			if (p.isAdjacent(defenderPos)){
					featureVectors[0] += 1.0;
				}
    		}
    	}
    	return featureVectors;
    }

    /**
     * DO NOT CHANGE THIS!yolo
     *
     * Prints the learning rate data described in the assignment. Do not modify this method.
     *
     * @param averageRewards List of cumulative average rewards from test episodes.
     */
    public void printTestData (List<Double> averageRewards) {
        System.out.println("");
        System.out.println("Games Played      Average Cumulative Reward");
        System.out.println("-------------     -------------------------");
        for (int i = 0; i < averageRewards.size(); i++) {
            String gamesPlayed = Integer.toString(10*i);
            String averageReward = String.format("%.2f", averageRewards.get(i));

            int numSpaces = "-------------     ".length() - gamesPlayed.length();
            StringBuffer spaceBuffer = new StringBuffer(numSpaces);
            for (int j = 0; j < numSpaces; j++) {
                spaceBuffer.append(" ");
            }
            System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
        }
        System.out.println("");
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will take your set of weights and save them to a file. Overwriting whatever file is
     * currently there. You will use this when training your agents. You will include th output of this function
     * from your trained agent with your submission.
     *
     * Look in the agent_weights folder for the output.
     *
     * @param weights Array of weights
     */
    public void saveWeights(Double[] weights) {
        File path = new File("agent_weights/weights.txt");
        // create the directories if they do not already exist
        path.getAbsoluteFile().getParentFile().mkdirs();

        try {
            // open a new file writer. Set append to false
            BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

            for (double weight : weights) {
                writer.write(String.format("%f\n", weight));
            }
            writer.flush();
            writer.close();
        } catch(IOException ex) {
            System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
        }
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will load the weights stored at agent_weights/weights.txt. The contents of this file
     * can be created using the saveWeights function. You will use this function if the load weights argument
     * of the agent is set to 1.
     *
     * @return The array of weights
     */
    public Double[] loadWeights() {
        File path = new File("agent_weights/weights.txt");
        if (!path.exists()) {
            System.err.println("Failed to load weights. File does not exist");
            return null;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line;
            List<Double> weights = new LinkedList<Double>();
            while((line = reader.readLine()) != null) {
                weights.add(Double.parseDouble(line));
            }
            reader.close();

            return weights.toArray(new Double[weights.size()]);
        } catch(IOException ex) {
            System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
        }
        return null;
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }
}
