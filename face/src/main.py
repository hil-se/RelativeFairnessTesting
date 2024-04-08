from rft import RelativeFairnessTesting
import time

def run():
    start = time.time()
    exp = RelativeFairnessTesting(rating_cols = ["P1", "P2", "P3", "Average"])
    exp.run()
    runtime = time.time()-start
    print(runtime)

if __name__ == "__main__":
    run()
