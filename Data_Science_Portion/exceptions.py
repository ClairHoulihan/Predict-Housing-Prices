class UnknownModelError(Exception):

    def __init__(self, model, message=" is not a supported regression model"):
        self.model = model
        self.message = model + message
        super().__init__(self.message)

class NoSelectorError(Exception):
    
    def __init__(self, message="Variable \"selector\" is not initialized"):
        self.message = message
        super().__init__(self.message)

class NoDataError(Exception):

    def __init__(self, message="No Data to form a model around"):
        self.message = message
        super().__init__(self.message)

class UnexpectedValueError(Exception):

    def __init__(self, value, expected_values):
        self.value = value

        self.expected_values = ""
        for val in expected_values:
            if val == expected_values[-1]:
                self.expected_values += "or %s" % (val)
            else:
                self.expected_values += "%s " % (val)
            

        self.message = "Received %s was expecting %s" % (self.value, self.expected_values)
        super().__init__(self.message)