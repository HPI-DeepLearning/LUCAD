import ConfigParser
import os
import logging


class LUCADConfig(object):
    def __init__(self, config_path = None, args = None):
        self.config_path = config_path
        self.args = args

        self._read_config()
        self._overwrite_with_args()

    def auto_section(self, value):
        sections = []
        for section in self.config.sections():
            values = {k: v for k, v in self.config.items(section)}
            if value in values:
                sections.append(section)
        if len(sections) == 1:
            return sections[0]
        else:
            raise RuntimeError("Auto section detection failed.")

    def str(self, value, section=None):
        if section is None:
            section = self.auto_section(value)
        return self.config.get(section, value)

    def int(self, value, section=None):
        if section is None:
            section = self.auto_section(value)
        return self.config.getint(section, value)

    def float(self, value, section=None):
        if section is None:
            section = self.auto_section(value)
        return self.config.getfloat(section, value)

    def bool(self, value, section=None):
        if section is None:
            section = self.auto_section(value)
        return self.config.getboolean(section, value)

    def write(self, fileobject):
        return self.config.write(fileobject)

    def _overwrite_with_args(self):
        if self.args is None:
            return

        args_dict = {k: v for k, v in vars(self.args).items() if v is not None}

        used = {name: False for name in args_dict}

        for section in self.config.sections():
            for name, value in self.config.items(section):
                if name in args_dict and args_dict[name] is not None:
                    self.config.set(section, name, str(args_dict[name]))
                    if used[name]:
                        logging.warning("Config option {section}.{name} was set, even though already set.").format(
                            section = section, name = name
                        )
                    used[name] = True
        if not all([val for _, val in used.items()]):
            logging.warning("Args not used in config: {args}".format(args=[key for key in used if not used[key]]))

    def _read_config(self):
        if self.config_path is None:
            # if config_path not set, use default config file
            self.config_path = os.path.join(os.getcwd(), "config", "ms-config.ini")

        self.config = ConfigParser.SafeConfigParser()
        logging.info("Reading config: %s" % self.config_path)
        self.config.read(self.config_path)
