from argparse import Action

class DictAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        result = {}
        for kv in values:
            key, value = kv.split('=')
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            elif value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            result[key] = value
        setattr(namespace, self.dest, result)