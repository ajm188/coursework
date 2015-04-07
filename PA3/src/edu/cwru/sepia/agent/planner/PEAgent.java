package edu.cwru.sepia.agent.planner;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.ActionType;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.agent.planner.actions.*;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Template;
import edu.cwru.sepia.environment.model.state.Unit;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

/**
 * This is an outline of the PEAgent. Implement the provided methods. You may add your own methods and members.
 */
public class PEAgent extends Agent {

    // The plan being executed
    private List<StripsAction> planList = null;

    // maps the real unit Ids to the plan's unit ids
    // when you're planning you won't know the true unit IDs that sepia assigns. So you'll use placeholders (1, 2, 3).
    // this maps those placeholders to the actual unit IDs.
    private Map<Integer, Integer> peasantIDMap;
    private int townhallId;
    private int peasantTemplateId;

    public PEAgent(int playernum, Stack<StripsAction> plan) {
        super(playernum);
        peasantIDMap = new HashMap<Integer, Integer>();
        // convert the plan from a stack to a list to make the parallelization easier/better
        // we promise we're not cheating
        this.planList = new ArrayList<StripsAction>();
        while (!plan.isEmpty()) {
        	planList.add(plan.pop());
        }
    }

    @Override
    public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {
        // gets the townhall ID and the peasant ID
    	/* 
    	 * our planner assumes the peasant id starts at 1 and goes up sequentially
    	 * use this as the key in the map
    	 */
    	int peasantCounter = 1;
        for(int unitId : stateView.getUnitIds(playernum)) {
            Unit.UnitView unit = stateView.getUnit(unitId);
            String unitType = unit.getTemplateView().getName().toLowerCase();
            if(unitType.equals("townhall")) {
                townhallId = unitId;
            } else if(unitType.equals("peasant")) {
                peasantIDMap.put(peasantCounter++, unitId);
            }
        }

        // Gets the peasant template ID. This is used when building a new peasant with the townhall
        for(Template.TemplateView templateView : stateView.getTemplates(playernum)) {
            if(templateView.getName().toLowerCase().equals("peasant")) {
                peasantTemplateId = templateView.getID();
                break;
            }
        }

        return middleStep(stateView, historyView);
    }

    /**
     * This is where you will read the provided plan and execute it. If your plan is correct then when the plan is empty
     * the scenario should end with a victory. If the scenario keeps running after you run out of actions to execute
     * then either your plan is incorrect or your execution of the plan has a bug.
     *
     * You can create a SEPIA deposit action with the following method
     * Action.createPrimitiveDeposit(int peasantId, Direction townhallDirection)
     *
     * You can create a SEPIA harvest action with the following method
     * Action.createPrimitiveGather(int peasantId, Direction resourceDirection)
     *
     * You can create a SEPIA build action with the following method
     * Action.createPrimitiveProduction(int townhallId, int peasantTemplateId)
     *
     * You can create a SEPIA move action with the following method
     * Action.createCompoundMove(int peasantId, int x, int y)
     *
     * these actions are stored in a mapping between the peasant unit ID executing the action and the action you created.
     *
     * For the compound actions you will need to check their progress and wait until they are complete before issuing
     * another action for that unit. If you issue an action before the compound action is complete then the peasant
     * will stop what it was doing and begin executing the new action.
     *
     * To check an action's progress you can call getCurrentDurativeAction on each UnitView. If the Action is null nothing
     * is being executed. If the action is not null then you should also call getCurrentDurativeProgress. If the value is less than
     * 1 then the action is still in progress.
     *
     * Also remember to check your plan's preconditions before executing!
     */
    @Override
    public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {
    	/* 
    	 * our planner assumes the peasant id starts at 1 and goes up sequentially
    	 * use this as the key in the map
    	 */
    	int peasantCounter = 1;
        for(int unitId : stateView.getUnitIds(playernum)) {
            Unit.UnitView unit = stateView.getUnit(unitId);
            String unitType = unit.getTemplateView().getName().toLowerCase();
            if(unitType.equals("townhall")) {
                townhallId = unitId;
            } else if(unitType.equals("peasant")) {
                peasantIDMap.put(peasantCounter++, unitId);
            }
        }
        
    	Map<Integer, Action> actions = new HashMap<Integer, Action>();
        // NOTE: We do not need to check our preconditions here, because they were already checked in
    	// the planning stage.
    	Set<Integer> busyUnits = new HashSet<Integer>();
    	
    	if (stateView.getTurnNumber() != 0) {
    		// check composite actions
    		Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
    		for (Integer unitID : actionResults.keySet()) {
    			ActionResult result = actionResults.get(unitID);
    			switch (result.getFeedback()) {
    			case COMPLETED:
    				// give the unit it's next action
    				break;
    			case INCOMPLETE:
    				// all good
    				busyUnits.add(unitID);
    				break;
    			case INCOMPLETEMAYBESTUCK:
    				// ??
    				break;
    			case FAILED:
    			case INVALIDCONTROLLER:
    			case INVALIDTYPE:
    			case INVALIDUNIT:
    				// #oops
    				return null;
    			}
    		}
    	}
    	
    	int i = 0;
    	while (i < planList.size()) {
    		Action nextAction = createSepiaAction(planList.get(i));
    		
    		if (nextAction == null) {
    			break;
    		}
    		
    		if (nextAction.getType() == ActionType.PRIMITIVEBUILD) {
    			if (busyUnits.isEmpty()) {
    				actions.put(nextAction.getUnitId(), nextAction);
    				planList.remove(i);
    			}
    			break;
    		}
    		
    		if (!busyUnits.contains(nextAction.getUnitId())) {
    			actions.put(nextAction.getUnitId(), nextAction);
    			busyUnits.add(nextAction.getUnitId());
    			planList.remove(i);
    		} else {
    			i++;
    		}
    	}
    	
        return actions;
    }

    /**
     * Returns a SEPIA version of the specified Strips Action.
     * @param action StripsAction
     * @return SEPIA representation of same action
     */
    private Action createSepiaAction(StripsAction action) {
    	if (action instanceof DepositGold){
    		DepositGold depositGold = (DepositGold) action;
    		Integer realPeasantID = peasantIDMap.get(depositGold.getPeasant().getID());
    		return realPeasantID == null ? null : Action.createCompoundDeposit(realPeasantID, depositGold.getTownHall().getID());
    	} else if (action instanceof DepositWood) {
    		DepositWood depositWood = (DepositWood) action;
    		Integer realPeasantID = peasantIDMap.get(depositWood.getPeasant().getID());
    		return realPeasantID == null ? null : Action.createCompoundDeposit(realPeasantID, depositWood.getTownHall().getID());
    	} else if (action instanceof HarvestGold) {
    		HarvestGold gold = (HarvestGold) action;
    		Integer realPeasantID = peasantIDMap.get(gold.getPeasant().getID());
    		return realPeasantID == null ? null : Action.createCompoundGather(realPeasantID, gold.getMine().getID());
    	} else if (action instanceof HarvestWood) {
    		HarvestWood wood = (HarvestWood) action;
    		Integer realPeasantID = peasantIDMap.get(wood.getPeasant().getID());
    		return realPeasantID == null ? null : Action.createCompoundGather(realPeasantID, wood.getForest().getID());
    	} else if (action instanceof BuildPeasant) {
    		return Action.createPrimitiveBuild(townhallId, peasantTemplateId);
    	} else {
    		//Ya dun fucked up.
        	return null;
    	}
    	
    }

    @Override
    public void terminalStep(State.StateView stateView, History.HistoryView historyView) {

    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }
}
