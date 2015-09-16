SEPIA_JAR=../jars/Sepia.jar
SEPIA_MAIN=edu.cwru.sepia.Main2 
JAVA_CMD=java -cp $(SEPIA_JAR) $(SEPIA_MAIN)

all: run

manual: data/ManualControlConfig.xml $(SEPIA_JAR) build
	$(JAVA_CMD) $<

run: data/ResourceCollectionConfig.xml $(SEPIA_JAR)  build
	$(JAVA_CMD) $<

config data/%.xml: $(SEPIA_JAR) build
	$(JAVA_CMD) $@

build:
