from flappyBird import FlappyBird

if __name__ == "__main__":
	flappy=FlappyBird()

	'''while flappy.maxScore<flappy.maxFinalScore:
		flappy.restart()
		flappy.run()
		flappy.geneticEvolution()'''
	flappy.loadLearnedNN()	
	while True:	
		flappy.restart()
		flappy.run()
	