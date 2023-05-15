from rft import RelativeFairnessTesting
import time

def run():
    start = time.time()
    exp = RelativeFairnessTesting()
    exp.run()
    runtime = time.time()-start
    print(runtime)


if __name__ == "__main__":
    run()
