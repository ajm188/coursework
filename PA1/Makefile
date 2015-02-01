JAVAC=javac
SEPIA_JAR=../jars/Sepia.jar
SEPIA_MAIN=edu.cwru.sepia.Main2
SEPIA_CP=-cp $(SEPIA_JAR)
SRC=src
JAVA_CMD=java $(SEPIA_CP):$(SRC) $(SEPIA_MAIN)
FULL_AGENT_PATH=$(SRC)/edu/cwru/sepia/agent/AstarAgent
FULL_ASTAR_NODE_PATH=$(SRC)/edu/cwru/sepia/environment/model/ExposedAStarNode
OUTPUT=$(FULL_AGENT_PATH).java $(FULL_ASTAR_NODE_PATH).java


all: run

run: agent
	$(JAVA_CMD) data/maze_8x8_config.xml

config %.xml: agent
	$(JAVA_CMD) data/$@
	
agent: $(OUTPUT)
	$(JAVAC) $(SEPIA_CP) $^

clean:
	rm -f $(FULL_AGENT_PATH)*.class $(FULL_ASTAR_NODE_PATH).class

