package edu.cwru.sepia.agent.planner.actions;

import java.io.IOException;
import java.util.List;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.agent.planner.GameState.Resource;
import edu.cwru.sepia.agent.planner.Position;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

public class HarvestWood implements StripsAction{

	private GameState.Peasant peasant;
	private GameState.Resource forest;
	//TODO: What do we do about the target? What the hell is the target? Townhall?
	private GameState.TownHall townhall;
//	private Position peasantPos;
//	private Position forestPos;
//	private Position targetPosition;
	
	public Position getPeasantPos(){
		return peasant.getPosition();		
	}
	
	//TODO: What if more than one forest?
	public Position getForestPos(GameState state){
		//TODO: Does this even belong here?
		 for (Resource resource : state.getResources()){
			if (resource.getType() == ResourceNode.Type.TREE){
				return resource.getPosition();
			}	 
		 }
		 return null;
	}
	
	public HarvestWood(GameState state){
		this.peasant = state.getPeasant();
		this.forest = state.getForest();
	}
	
	public boolean preconditionsMet(GameState state) {
		Unit.UnitView peasantView = state.getPeasantView();

		if (peasantView == null) {
			return false;
		}

		if (state.getStateView().isResourceAt(forestPos.x, forestPos.y)){
			ResourceNode.ResourceView forest = state.getStateView().getResourceNode(state.getStateView().resourceAt(forestPos.x, forestPos.y));
			Position peasantPosition = new Position(peasantView.getXPosition(), peasantView.getYPosition());
			
			List<Position> forestPosAdjacents = forestPos.getAdjacentPositions();
			boolean adjEmpty = false;
			for (Position adjacentPosition : forestPosAdjacents) {
				if (!(state.getStateView().isResourceAt(adjacentPosition.x, adjacentPosition.y) || state.getStateView().isUnitAt(adjacentPosition.x, adjacentPosition.y))) {
					this.targetPosition = adjacentPosition;
					adjEmpty = true;
					break;
				}
			}
			
			return peasantPosition.equals(peasantPos) &&
					adjEmpty &&
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

		state.transportUnit(peasant, targetPosition.x, targetPosition.y); // move the peasant next to the townhall
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
