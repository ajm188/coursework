JAVAC=javac
SEPIA_JAR=../jars/Sepia.jar
SEPIA_MAIN=edu.cwru.sepia.Main2
SEPIA_CP=-cp $(SEPIA_JAR)
JAVA_CMD=java $(SEPIA_CP) $(SEPIA_MAIN)
SRC=src
FULL_AGENT_PATH=$(SRC)/edu/cwru/sepia/agent/AstarAgent
OUTPUT=$(FULL_AGENT_PATH).java


all: run

run:
	$(JAVA_CMD) data/maze_8x8_config.xml

config %.xml: agent
	$(JAVA_CMD) data/$@
	
agent: $(OUTPUT)
	$(JAVAC) $(SEPIA_CP) $<

clean:
	rm -f $(FULL_AGENT_PATH)*.class

