package edu.cwru.sepia.agent.minimax;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionType;
import edu.cwru.sepia.action.DirectedAction;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.UnitTemplate;
import edu.cwru.sepia.environment.model.state.UnitTemplate.UnitTemplateView;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.util.Direction;

import java.util.*;
import java.io.IOException;

/**
 * This class stores all of the information the agent
 * needs to know about the state of the game. For example this
 * might include things like footmen HP and positions.
 *
 * Add any information or methods you would like to this class,
 * but do not delete or change the signatures of the provided methods.
 */
public class GameState {

    private int xExtent;
    private int yExtent;

    private boolean isMax;

    private Double utility;

    private List<ResourceNode.ResourceView> resourceNodes;

    private List<UnitView> footmen;
    private List<UnitView> archers;

    private State.StateView state;

    private List<GameStateChild> children;
    /**
     * You will implement this constructor. It will
     * extract all of the needed state information from the built in
     * SEPIA state view.
     *
     * You may find the following state methods useful:
     *
     * state.getXExtent() and state.getYExtent(): get the map dimensions
     * state.getAllResourceIDs(): returns all of the obstacles in the map
     * state.getResourceNode(Integer resourceID): Return a ResourceView for the given ID
     *
     * For a given ResourceView you can query the position using
     * resource.getXPosition() and resource.getYPosition()
     *
     * For a given unit you will need to find the attack damage, range and max HP
     * unitView.getTemplateView().getRange(): This gives you the attack range
     * unitView.getTemplateView().getBasicAttack(): The amount of damage this unit deals
     * unitView.getTemplateView().getBaseHealth(): The maximum amount of health of this unit
     *
     * @param state Current state of the episode
     */
    public GameState(State.StateView state) {
        extractInfoFromState(state);
    }

    public GameState(State.StateView state, boolean isMax)
    {
        this.isMax = isMax;
        extractInfoFromState(state);
    }

    private void extractInfoFromState(State.StateView state)
    {
        xExtent = state.getXExtent();
        yExtent = state.getYExtent();

        resourceNodes = state.getAllResourceNodes();

        footmen = new ArrayList<UnitView>();
        archers = new ArrayList<UnitView>();
        for (UnitView unit : state.getAllUnits()) {
            if (unit.getTemplateView().getName().equals("Footman")) {
                footmen.add(unit);
            } else if (unit.getTemplateView().getName().equals("Archer")) {
                archers.add(unit);
            }
        }

        this.state = state;

        children = null;

        utility = null;
    }

    /**
     * You will implement this function.
     *
     * You should use weighted linear combination of features.
     * The features may be primitives from the state (such as hp of a unit)
     * or they may be higher level summaries of information from the state such
     * as distance to a specific location. Come up with whatever features you think
     * are useful and weight them appropriately.
     *
     * It is recommended that you start simple until you have your algorithm working. Then watch
     * your agent play and try to add features that correct mistakes it makes. However, remember that
     * your features should be as fast as possible to compute. If the features are slow then you will be
     * able to do less plys in a turn.
     *
     * Add a good comment about what is in your utility and why you chose those features.
     *
     * @return The weighted linear combination of the features
     */
    public double getUtility() {
        if (utility != null) {
            return utility;
        }
        /*
         * Features we want:
         *  - health of the footmen (positive)
         *  - health of the archers (negative)
         *  - number of footmen (positive)
         *  - number of archers (negative)
         *  - distance from footmen to archers (negative) --> not sure if this should be positive or negative
         * Adjust the weights as we test
         */
        double footmanToArcherDistance = 0.0;
        for (UnitView footman : footmen) {
            double currentMin = Double.POSITIVE_INFINITY;
            for (UnitView archer : archers) {
                currentMin = Math.min(currentMin, chebyshev(footman, archer));
            }
            footmanToArcherDistance += currentMin;
        }

        utility = 1 * getTotalHealth(footmen) + (-1) * getTotalHealth(archers)
            + 1 * footmen.size() + (-1) * archers.size() + 1 * footmanToArcherDistance;

        return utility;
    }

    private double chebyshev(UnitView unit1, UnitView unit2)
    {
        return Math.max(unit2.getXPosition() - unit1.getXPosition(), unit2.getYPosition() - unit1.getYPosition());
    }

    private int getTotalHealth(List<UnitView> units)
    {
        int health = 0;
        for (UnitView unit : units)
        {
            health += unit.getHP();
        }
        return health;
    }

    /**
     * You will implement this function.
     *
     * This will return a list of GameStateChild objects. You will generate all of the possible
     * actions in a step and then determine the resulting game state from that action. These are your GameStateChildren.
     *
     * You may find it useful to iterate over all the different directions in SEPIA.
     *
     * for(Direction direction : Directions.values())
     *
     * To get the resulting position from a move in that direction you can do the following
     * x += direction.xComponent()
     * y += direction.yComponent()
     *
     * @return All possible actions and their associated resulting game state
     */
    public List<GameStateChild> getChildren() {
        if (children != null)
        {
            return children; // do some caching for speed
        }

        //depending on whose turn it is, collect all possible actions
        List<UnitView> desiredList = isMax ? footmen : archers;
        Map<Integer,List<Action>> unitActions = new HashMap<Integer, List<Action>>();
        for (UnitView unit : desiredList) {
            unitActions.put(unit.getID(), getAllActions(unit));
        }

        /*
         * Ok, here we go. We need to get all of the pairwise combinations of the
         * actions of the units. If there is only one unit, great! This is really
         * easy then. Just create a list of maps, each of which maps the single
         * unit id to each of the actions that unit can make in this state.
         *
         * If there are two units that can move, then it gets more complicated.
         * If the second unit has k moves, then each of the first unit's move maps
         * needs to be copied k times, so that each of the first unit's moves can
         * be paired exactly once with each of the second unit's moves. We also
         * do some error checking to ensure that the two units don't do something
         * that would be simultaneously illegal (like trying to move to the same square).
         */
        int firstUnitID = desiredList.get(0).getID();
        Integer secondUnitID = desiredList.size() > 1 ? desiredList.get(1).getID() : null;
        // if there is no second unit, we want to copy everything once (which is basically not copying at all)
        int timesToCopy = secondUnitID != null ? unitActions.get(secondUnitID.intValue()).size() : 1;

        List<Map<Integer, Action>> actionsList = new ArrayList<Map<Integer, Action>>();

        // collect all of the actions for the first unit, copying them if necessary (see the above block comment)
	// Liberatore is disappointed in this McCabe's complexity.
	if (secondUnitID==null){
		//Then there is only one unit to iterate over
		for (Action action : unitActions.get(firstUnitID)){
			Map<Integer, Action> temp = new HashMap<Integer, Action>();
			temp.put(firstUnitID, action);
			actionsList.add(temp);
		}
	} else {
		for (Action action1 : unitActions.get(firstUnitID)){
			for (Action action2 : unitActions.get(secondUnitID)){
				if (isLegal(action1, action2)) {
					Map<Integer, Action> temp = new HashMap<Integer, Action>();
					temp.put(firstUnitID, action1);
					temp.put(secondUnitID, action2);
					actionsList.add(temp);
				}
			}	
		}
	}


        List<GameStateChild> children = new ArrayList<GameStateChild>();
        for (Map<Integer, Action> actions : actionsList)
        {
            // apply the set of actions and return the GameStateChild that results from these actions
            children.add(applyActions(actions));
        }

        this.children = children;

        return this.children;
    }

    private GameStateChild applyActions(Map<Integer, Action> actions)
    {
        State stateClone;
        try
        {
            stateClone = this.state.getStateCreator().createState();
        }
        catch (IOException e)
        {
            return null;
        }
        for (Integer unitID : actions.keySet())
        {
            Action action = actions.get(unitID);
            switch (action.getType())
            {
                case PRIMITIVEMOVE:
                    DirectedAction move = (DirectedAction) action;
                    stateClone.moveUnit(stateClone.getUnit(unitID), move.getDirection());
                    break;
                case PRIMITIVEATTACK:
                    TargetedAction attack = (TargetedAction) action;
                    Unit attacker = stateClone.getUnit(unitID);
                    Unit defender = stateClone.getUnit(attack.getTargetId()); // WHY IS THIS ONE SPELLED DIFFERENTLY??? EITHER ALWAYS DO "ID" OR ALWAYS DO "Id" NOT BOTH!!!
                    defender.setHP(defender.getCurrentHealth() - attacker.getTemplate().getBasicAttack());
                    break;
                default:
                    // this really should never happen but it never hurts to be careful
                    break;
            }
        }
        return new GameStateChild(actions, new GameState(stateClone.getView(0), !isMax));
    }

    private List<Action> getAllActions(UnitView unit){
        List<Action> allActions = new ArrayList<Action>();
	
        // get all move actions
        for(Direction direction: Direction.values()){
            if (direction.xComponent() == 0 && direction.yComponent() == 0){
                continue;
            }
            if (isLegal(unit, direction)){
                allActions.add(Action.createPrimitiveMove(unit.getID(), direction));
            }
        }
	
        // get all attacking actions
        UnitTemplateView unitTemplateView = unit.getTemplateView();
        int upperBound = unitTemplateView.getRange();
        int lowerBound = -upperBound;
        for (int xAdj = lowerBound; xAdj <= upperBound; xAdj++) {
            for (int yAdj = lowerBound; yAdj <= upperBound; yAdj++) {
                if (xAdj == 0 && yAdj == 0) {
                    continue;
                }

                int newX = unit.getXPosition() + xAdj;
                int newY = unit.getYPosition() + yAdj;
                Integer enemyUnitID = state.unitAt(newX, newY);
                if (enemyUnitID == null) {
                    // no unit here, next iteration
                    continue;
                }

                Unit.UnitView enemy = state.getUnit(enemyUnitID);
                if (enemy.getTemplateView().getCharacter() == unit.getTemplateView().getCharacter()) {
                    // this is not actually an enemy, next iteration
                    continue;
                }

                allActions.add(Action.createPrimitiveAttack(unit.getID(), enemyUnitID));
            }
        }

        return allActions;
    }

     private boolean isLegal(UnitView unit, Direction direction){
        int x = unit.getXPosition() + direction.xComponent();
        int y = unit.getYPosition() + direction.yComponent();
     	boolean inBounds = x < xExtent && x >= 0 && y < yExtent && y >= 0;
        for (ResourceNode.ResourceView resourceView : resourceNodes){
            if (!inBounds){
                return false;
            }
            inBounds = inBounds || resourceView.getXPosition() != x && resourceView.getYPosition() != y;	
        }
        return inBounds;
     }

     /**
      * Are these two actions legal (together)?
      * The only actions that can conflict are move actions that attempt to move to the same square.
      */
     private boolean isLegal(Action action1, Action action2)
     {
         ActionType primitiveMove = ActionType.PRIMITIVEMOVE;
         if (action1.getType() != primitiveMove || action2.getType() != primitiveMove)
         {
             return false;
         }

         DirectedAction moveAction1 = (DirectedAction) action1;
         DirectedAction moveAction2 = (DirectedAction) action2;

         Unit.UnitView unit1 = state.getUnit(moveAction1.getUnitId());
         Unit.UnitView unit2 = state.getUnit(moveAction2.getUnitId());

         Direction direction1 = moveAction1.getDirection();
         Direction direction2 = moveAction2.getDirection();

         int x1 = unit1.getXPosition() + direction1.xComponent();
         int y1 = unit1.getYPosition() + direction1.yComponent();

         int x2 = unit2.getXPosition() + direction2.xComponent();
         int y2 = unit2.getYPosition() + direction2.yComponent();

         return (x1 != x2 || y1 != y2);
     }
}
