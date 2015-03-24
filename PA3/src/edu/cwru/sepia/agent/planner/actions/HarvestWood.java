package edu.cwru.sepia.agent.planner.actions;

import java.io.IOException;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.agent.planner.Position;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

public class HarvestWood implements StripsAction{

	private Position peasantPos;
	private Position forestPos;
	
	public HarvestWood(Position peasantPos, Position forestPos){
		this.peasantPos = peasantPos;
		this.forestPos = forestPos;
	}
	
	public boolean preconditionsMet(GameState state) {
		Unit.UnitView peasant = state.getUnits().get(0);
		if (state.getStateView().isResourceAt(forestPos.x, forestPos.y)){
			ResourceNode.ResourceView forest = state.getStateView().getResourceNode(state.getStateView().resourceAt(forestPos.x, forestPos.y));
			return forest.getType() == ResourceNode.Type.TREE &&	
					forest.getAmountRemaining() >= 100 && 
					peasant.getCargoAmount() == 0;
		} else {
			return false;
		}
	}

	public GameState apply(GameState gameState) {
		State state;
		try {
			state = gameState.getStateView().getStateCreator().createState();
		} catch (IOException e) {
			return null;
		}
		
		Unit peasant = state.getUnit(gameState.getStateView().unitAt(peasantPos.x, peasantPos.y));
		ResourceNode forest = state.resourceAt(forestPos.x, forestPos.y);

		peasant.setCargo(ResourceType.WOOD, 100);
		forest.reduceAmountRemaining(100);
		
		return new GameState(state.getView(gameState.getPlayerNum()),
				gameState.getPlayerNum(),
				gameState.getRequiredGold(),
				gameState.getRequiredWood(),
				gameState.getBuildPeasants());	
	}
}
