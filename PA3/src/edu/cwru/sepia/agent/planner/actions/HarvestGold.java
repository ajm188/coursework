package edu.cwru.sepia.agent.planner.actions;

import java.io.IOException;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.agent.planner.Position;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

public class HarvestGold implements StripsAction{

	private Position peasantPos;
	private Position minePos;
	
	public HarvestGold(Position peasantPos, Position minePos){
		this.peasantPos = peasantPos;
		this.minePos = minePos;
	}
	
	public boolean preconditionsMet(GameState state) {
		Unit.UnitView peasantView = state.getPeasantView();

		if (peasantView == null) {
			return false;
		}

		if (state.getStateView().isResourceAt(minePos.x, minePos.y)){
			ResourceNode.ResourceView mine = state.getStateView().getResourceNode(state.getStateView().resourceAt(minePos.x, minePos.y));
			Position peasantPosition = new Position(peasantView.getXPosition(), peasantView.getYPosition());
			
			return peasantPosition.equals(peasantPos) &&
					peasantPos.isAdjacent(minePos) &&
					mine.getType() == ResourceNode.Type.GOLD_MINE &&	
					mine.getAmountRemaining() >= 100 && 
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
		ResourceNode mine = state.resourceAt(minePos.x, minePos.y);

		peasant.setCargo(ResourceType.GOLD, 100);
		mine.reduceAmountRemaining(100);
		
		return new GameState(state.getView(gameState.getPlayerNum()),
				gameState.getPlayerNum(),
				gameState.getRequiredGold(),
				gameState.getRequiredWood(),
				gameState.getBuildPeasants(),
				gameState,
				this);	
	}
	
	

}
