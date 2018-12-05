import functools

class BaseStageProcess(object):
    def __init__(self):
        self.nii = None
        self.slices = None
        self.para = self.init_para()
        self.bind_default_para()

    @staticmethod
    def error_retry(func, exception_action, max_try_time=5):
        try_time = 0
        while try_time < max_try_time:
            try:
                func()
                break
            except Exception as e:
                exception_action(e)
                try_time += 1

    def default_para(self, func):
        @functools.wraps(func)
        def _wrap(*args, **kwargs):
            func_name = func.__code__.co_name
            args_list = list(zip(func.__code__.co_varnames[1:], args))
            kwargs_list = list(kwargs.items())
            no_save_para = {}
            for key, value in args_list + kwargs_list:
                if key in ['img', 'nii', 'component', 'mask', 'test', 'type_', 'point', 'stage']:
                    no_save_para[key] = value
                else:
                    self.para[func_name][key] = value
            return func(**no_save_para, **self.para[func_name])
        return _wrap

    def bind_default_para(self):
        for func_name in self.para:
            func = getattr(self, func_name)
            setattr(self, func_name, self.default_para(func))

    def run(self, nii, stage):
        raise NotImplementedError()

    def show(self):
        raise NotImplementedError()

    def init_para(self):
        raise NotImplementedError()
