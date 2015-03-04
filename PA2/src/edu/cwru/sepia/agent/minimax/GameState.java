package edu.cwru.sepia.agent.minimax;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionType;
import edu.cwru.sepia.action.DirectedAction;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.util.Direction;

import java.util.*;

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

    private boolean max;

    private Double utility;

    private List<ResourceNode.ResourceView> resourceNodes;

    private List<UnitView> footmen;
    private List<UnitView> archers;
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
        xExtent = state.getXExtent();
        yExtent = state.getYExtent();

        resourceNodes = state.getAllResourceNodes();

        footmen = new ArrayList<UnitView>();
        archers = new ArrayList<UnitView>();
        for (UnitView unit : state.getAllUnits()) {
            if (unit.getTemplateView().getName().equals("footman")) {
                footmen.add(unit);
            } else if (unit.getTemplateView().getName().equals("archer")) {
                archers.add(unit);
            }
        }

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
        if (utility == null) {
            utility = 0.0; // do semthing here ...
        }
        return utility;
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
        //depending on whose turn it is, collect all possible actions
        List<UnitView> desiredList = isMax ? footmen : archers;
        Map<Integer,List<Action>> unitActions = new HashMap<Integer, List<Action>>();
        for (UnitView unit : desiredList) {
            unitActions.put(unit.getID(), getAllActions(unit));
        }	

        return null;
    }

    private List<Action> getAllActions(UnitView unit){
        List<Action> allActions = new ArrayList<Action>();
	
        for(Direction direction: Directions.values()){
            if (direction.xComponent() == 0 && direction.yComponent() == 0){
                continue;
            }
            if (isLegal(direction)){
                allActions.add(Action.createPrimitiveMove(unit.getID(), direction));
            }
        }
	
        //for each unit, get units in range	

        return allActions;
    }

     private boolean isLegal(UnitView unit, Direction direction){
        int x = unit.getXPosition() + direction.xComponent;
        int y = unit.getYPosition() + direction.yComponent;
     	boolean inBounds = x < xExtent && x >= 0 && y < yExtent && y >= 0;
        for (ResourceNode.ResourceView resourceView : resourceNodes){
            if (!inBounds){
                return false;
            }
            inBounds = inBounds || resourceView.getXPosition() != x && resourceView.getYPosition != y;	
        }
        return inBounds;
     }
}
