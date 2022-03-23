from db4ml import execute
from {{cookiecutter.package}}.common import Job


class SampleJob(Job):
    def launch(self):
        self.logger.info("Launching sample job")

        execute("{{cookiecutter.model}}_train", self.spark, self.config_path)
        execute("{{cookiecutter.model}}_scoring_pandas", self.spark, self.config_path)

        self.logger.info("Sample job finished!")


if __name__ == "__main__":
    job = SampleJob()
    job.launch()
