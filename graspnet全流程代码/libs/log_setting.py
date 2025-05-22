import os

import logging
from logging.handlers import TimedRotatingFileHandler

class CommonLog(object):
    """
    日志记录
    """

    def __init__(self, logger, logname='web-log'):
        self.logname = os.path.join(os.path.dirname(os.path.abspath(__file__)), '%s' % logname)
        self.logger = logger
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # 禁止使用logger对象parent的处理器
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    def __console(self, level, message):
        # 创建一个FileHandler，用于写到本地

        fh = TimedRotatingFileHandler(self.logname, when='MIDNIGHT', interval=1, encoding='utf-8')
        # fh = logging.FileHandler(self.logname, 'a', encoding='utf-8')
        fh.suffix = '%Y-%m-%d.log'
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

        # 创建一个StreamHandler,用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message, exc_info=1)  # 显示错误栈
            # self.logger.error(message)

        elif level == 'error_':
            self.logger.error(message)  # 不显示错误栈


        # 这两行代码是为了避免日志输出重复问题
        self.logger.removeHandler(ch)
        self.logger.removeHandler(fh)
        # 关闭打开的文件
        fh.close()

    def debug(self, message):
        self.__console('debug', message)

    def info(self, message):
        self.__console('info', message)

    def warning(self, message):
        self.__console('warning', message)

    def error(self, message):
        self.__console('error', message)

    def error_(self, message):
        self.__console('error_', message)
