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
		Unit.UnitView peasantView = state.getPeasantView();

		if (peasantView == null) {
			return false;
		}

		if (state.getStateView().isResourceAt(forestPos.x, forestPos.y)){
			ResourceNode.ResourceView forest = state.getStateView().getResourceNode(state.getStateView().resourceAt(forestPos.x, forestPos.y));
			Position peasantPosition = new Position(peasantView.getXPosition(), peasantView.getYPosition());
			
			return peasantPosition.equals(peasantPos) &&
					peasantPos.isAdjacent(forestPos) &&
					forest.getType() == ResourceNode.Type.TREE &&	
					forest.getAmountRemaining() >= 100 && 
					peasantView.getCargoAmount() == 0;
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
				gameState.getBuildPeasants(),
				gameState,
				this);	
	}
}
