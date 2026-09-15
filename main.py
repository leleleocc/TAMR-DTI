from src.core.main import main, time, datetime


if __name__ == "__main__":
    print(f"start time: {datetime.now()}")
    start = time()
    result = main()
    end = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(end - start, 2)}s, ")
