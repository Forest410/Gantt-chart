from matplotlib.colors import to_rgba
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.dates as mdates
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

#########################
# FUNCIONES SECUNDARIAS #
#########################


def importamos_ficheros():
    df = pd.read_csv("input/datos.csv", sep="\t")
    return df


def configuracion_grafica(df, titulo='Integrated Planning', size=(15, 10), pdf=True, todayis='15/03/2021', sdate='01/06/2020', fdate='01/01/2022', btx_start='01/07/2021', btx_end='01/03/2022'):
    df['Start Date'] = pd.to_datetime(df['Start Date'], format='%d/%m/%Y')
    df['Finish Date'] = pd.to_datetime(df['Finish Date'], format='%d/%m/%Y')
    df['BL Start'] = pd.to_datetime(df['BL Start'], format='%d/%m/%Y')
    df['BL Finish'] = pd.to_datetime(df['BL Finish'], format='%d/%m/%Y')
    df.sort_values(by=["Project", "Subproject", "Start Date", "Finish Date"])

    data_list = df.T.to_dict().values()

    today_date = datetime.strptime(todayis, "%d/%m/%Y").date()
    today = datetime(today_date.year, today_date.month, today_date.day)

    res = {}
    packages = []
    index = 0

    start_time = datetime.strptime(sdate, "%d/%m/%Y").date()
    end_time = datetime.strptime(fdate, "%d/%m/%Y").date()

    for item in data_list:
        color = ""
        if item['% Completed'] == '100%':
            color = 'green'
        else:
            color = 'blue'
        if item['Start Date'].date() < start_time or item['Finish Date'].date() > end_time:
            continue
        temp = {
            "index": index,
            "label1": item['Project'],
            "label": item['Subproject'],
            "name": item['Name'],
            "start": item['Start Date'].date(),
            'end': item['Finish Date'].date(),
            "color": color,
            "milestones": [],
            "bl_milestones": [],
            "bl_start": item['Start Date'].date(),
            "bl_finish": item['Start Date'].date(),
            "bl_flg": False
        }
        if item['Start Date'] == item['Finish Date']:
            temp['milestones'].append(item['Finish Date'].date())
        if item['Delay?'] == "YES":
            temp['bl_flg'] = True
            if item['BL Start'] == item['BL Finish']:
                temp['bl_milestones'].append(item['BL Finish'].date())
                temp['bl_start'] = item['BL Start'].date()
                temp['bl_finish'] = item['BL Finish'].date()
            else:
                temp['bl_start'] = item['BL Start'].date()
                temp['bl_finish'] = item['BL Finish'].date()
        packages.append(temp)
        index += 1

    res["packages"] = packages
    res["today"] = today
    res['title'] = titulo
    res['size'] = size
    res['start_time'] = start_time
    res['end_time'] = end_time
    res['pdf'] = pdf
    res['btx_start'] = btx_start
    res['btx_end'] = btx_end
    return res


def grafica(configuracion):
    configuracion = Gantt(configuracion)
    configuracion.render()
    # configuracion.show()
    return configuracion.show()

###########################################
# FUNCIONES, OBJETOS Y CLASES PRINCIPALES #
###########################################


class Package():
    def __init__(self, pkg):
        self.lable1 = pkg['label1']
        self.label = pkg['label']
        self.name = pkg['name']
        self.start = pkg['start']
        self.end = pkg['end']
        self.index = pkg['index']
        self.color = pkg['color']
        self.bl_start = pkg['bl_start']
        self.bl_finish = pkg['bl_finish']
        self.bl_flg = pkg['bl_flg']
        try:
            self.milestones = pkg['milestones']
            self.bl_milestones = pkg['bl_milestones']
        except KeyError:
            pass
        try:
            self.legend = pkg['legend']
        except KeyError:
            self.legend = None


class Gantt():
    def __init__(self, dataset):
        self.dataFile = dataset
        self.packages = []
        self.labels = []
        self.xticks = []
        self.projects = []            # yaxis. labels
        self.ax1_month = []           # month axis
        self.main_Lables = []
        self.main_LablesPos = []
        self.size = []
        self._loadData()
        self._procData()

    def _loadData(self):
        register_matplotlib_converters()
        self.size = self.dataFile['size']
        self.title = self.dataFile['title']
        self.pdf = self.dataFile['pdf']
        for pkg in self.dataFile['packages']:
            self.packages.append(Package(pkg))
        self.labels = [pkg['label'] for pkg in self.dataFile['packages']]
        self.main_Lables = [pkg['label1'] for pkg in self.dataFile['packages']]
        self.start_time = self.dataFile['start_time']
        self.end_time = self.dataFile['end_time']
        self.today = self.dataFile['today']
        self.btx_start = self.dataFile['btx_start']
        self.btx_end = self.dataFile['btx_end']
        cnt = 0
        for idx in range(len(self.labels)):
            if idx == 0:
                cnt += 1
                continue
            else:
                if self.labels[idx] == self.labels[idx-1]:
                    cnt += 1
                else:
                    self.projects.append(len(self.labels)-idx)
                    cnt = 1
        self.milestones = {}
        self.bl_milestones = {}
        for pkg in self.packages:
            try:
                self.milestones[pkg.index] = pkg.milestones
                self.bl_milestones[pkg.index] = pkg.bl_milestones
            except AttributeError:
                pass
        try:
            self.xlabel = self.dataFile['xlabel']
        except KeyError:
            self.xlabel = ""

    def _procData(self):
        self.nPackages = len(self.labels)
        self.start = [None] * self.nPackages
        self.end = [None] * self.nPackages
        self.blstart = [None] * self.nPackages
        self.blfinish = [None] * self.nPackages
        for pkg in self.packages:
            self.start[pkg.index] = pkg.start
            self.end[pkg.index] = pkg.end
            if pkg.bl_flg == True:
                self.blstart[pkg.index] = pkg.bl_start
                self.blfinish[pkg.index] = pkg.bl_finish
            else:
                self.blstart[pkg.index] = pkg.start
                self.blfinish[pkg.index] = pkg.start
        self.durations = map(lambda x, y: (x-y).days, self.end, self.start)
        self.bldurations = map(lambda x, y: (x-y).days,
                               self.blfinish, self.blstart)
        self.yPos = np.arange(self.nPackages, 0, -1)

    def month_format(self):
        # Definimos la linea vertical para indicar la fecha actual => self.today viene del dataframe `res` generado por las funciones secundarias
        # => Ahí es donde editamos la fecha para establecer el today date
        plt.axvline(self.today, color="red", label=str(self.today))
        plt.rcParams['figure.figsize'] = (100, 100)
        min_year = self.start_time.year
        max_year = self.end_time.year
        min_month = self.start_time.month
        max_month = self.end_time.month
        month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                      'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.ax1_year = []
        for year in range(min_year, max_year + 1):
            self.ax1_year.append(year)
            for month in range(1, 13):
                if year == min_year and month <= min_month:
                    continue
                elif year == max_year and month > max_month:
                    break
                self.xticks.append(datetime.date(
                    datetime(year, month, 1)))
                self.ax1_month.append(month_list[month-1] + "-" + str(year))
                # GRID horizontal line format
                #plt.axvline(datetime.date(datetime(year, month, 1)),color='gray', linestyle='--')

    def format(self):
        plt.tick_params(
            axis='both',    # format x and y
            # which='both',   # major and minor ticks affected
            direction="out",
            length=4,
            width=1,
        )
        self.month_format()
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        years_fmt = mdates.DateFormatter('%M/%Y')
        ax = plt.gca()
        ax.axes.get_yaxis().set_visible(False)
        today = datetime.date(datetime.now())
        rects = ax.patches

        # Make labels.
        labels = []
        for item in self.packages:
            # LABELS para Milestone y Tasks
            if item.start != item.end:
                labels.append(item.name + " | " + item.start.strftime('%d-%m-%Y') +
                              "->" + item.end.strftime('%d-%m-%Y'))
            else:
                labels.append(item.name + " | " +
                              item.end.strftime('%d-%m-%Y'))
        idx = 0

        # Display labels to graph: height is position of label when print it.
        # we can set position of label to position of graph + offset. So you can control offset for that.
        for rect, label in zip(rects, labels):
            # Parametro offset para los labels sobre task/milestone
            height = self.yPos[idx] + .25
            ax.text(self.packages[idx].start, height,
                    label, ha='left', va='bottom')
            # IMPORTANTE: La Posicion para las labels Milestone y Tasks. Range valores para `ha` = 'center', 'right', 'left'. Rango valores para `va`='top', 'bottom'.
            # NOTA 1: Cuando indicamos 'left' => las labels se van a la derecha y viceversa.
            # NOTA 2: Cuando indicamos 'bottom' => se suben arriba del milestone/task y viceversa cuando indicamos `top`
            # Para cambiar el tamaño de letra de las labels de MS/Task => fontsize=15
            # Para poner negrita =>  fontweight='bold'
            # Para poner cursiva =>  style='italic'
            # Para darle color al texto => color='green' o color='valor-hexadecimal'
            idx += 1
        # plt.xlim(min(self.start) - timedelta(days=10), max(max(self.end), today + timedelta(days=10)), timedelta(days=7))
        plt.xlim(self.start_time, self.end_time, timedelta(days=7))

        # Establecemos los limites superiores e inferiores para que los Milestones/Tareas se vean perfectamente
        plt.ylim(0.5, self.nPackages + 1)

        # plt.yticks(self.yPos, self.labels, rotation=90) # Y-LABELS Valores and Rotation
        # Para cambiar el tamaño de letra de las YLABELS => fontsize=15
        # Para poner negrita =>  fontweight='bold'
        # Para poner cursiva =>  style='italic'
        # Para darle color al texto => color='green' o color='valor-hexadecimal'

        # make grid about every project
        for item in self.projects:
            # GRID horizontal line format
            plt.axhline(y=item + .65, color="#A9A9A9", linestyle="-")

        # Title setup
        plt.title(self.title, x=0.35, weight='bold')       # EL TITULO
        # Para cambiar el tamaño de letra del título => fontsize=15
        # Para poner negrita =>  fontweight='bold'
        # Para poner cursiva =>  style='italic'
        # Para darle color al texto => color='green' o color='valor-hexadecimal'
        plt.margins(0.2)

        if self.xticks:

            # X-LABELS Valores and Rotation
            plt.xticks(self.xticks, self.ax1_month, rotation=40)
            # Para cambiar el tamaño de letra de las XLABELS => fontsize=15
            # Para poner negrita =>  fontweight='bold'
            # Para poner cursiva =>  style='italic'
            # Para darle color al texto => color='green' o color='valor-hexadecimal'

    # add milestones to graph
    def add_milestones(self):
        if not self.milestones:
            return
        x = []
        y = []
        colors = []
        for key in self.milestones.keys():
            for value in self.milestones[key]:
                colors.append(self.packages[key].color)
                y += [self.yPos[key]]
                x += [value]

        # Milestone Shape, fill, border color. El color del fill es `colors` esta parametrizado y el dato sale de las funciones secundarias
        plt.scatter(x, y, s=120, marker="D", color=colors,
                    edgecolor="black", zorder=0)

    # add milestones when delay = "YES"
    def add_blmilestones(self):
        if not self.milestones:
            return
        x = []
        y = []
        colors = []
        for key in self.bl_milestones.keys():
            for value in self.bl_milestones[key]:
                colors.append(self.packages[key].color)
                y += [self.yPos[key]]
                x += [value]
                if self.packages[key].bl_flg == True:
                    # Baseline Milestone Shape, fill, border color
                    plt.scatter(
                        value, self.yPos[key], s=120, marker="D", color="#A9A9A9", edgecolor="black", zorder=3)

    def add_legend(self):
        cnt = 0
        for pkg in self.packages:
            if pkg.legend:
                cnt += 1
                idx = self.labels.index(pkg.label)
                self.barlist[idx].set_label(pkg.legend)

        if cnt > 0:
            self.legend = self.ax.legend(
                shadow=False, ncol=3, fontsize="large")

    def add_dashline(self):
        idx = 0
        for pkg in self.packages:
            if pkg.bl_flg == True:
                # Linea Discontinua Roja Baseline vs Planned Date
                if len(pkg.milestones) == 0 and len(pkg.bl_milestones) == 0:
                    plt.plot([pkg.start, pkg.bl_finish], [
                        self.yPos[idx]-.2, self.yPos[idx]-.2], "--", color="red")
                else:
                    plt.plot([pkg.start, pkg.bl_finish], [
                        self.yPos[idx]-.1, self.yPos[idx]-.1], "--", color="red")
            idx += 1

    def annotate_yrange(self, ymin, ymax,
                        label=None, fontsize=12,
                        offset=-0.1,
                        width=-0.1,
                        text_kwargs={'rotation': 'horizontal'},
                        ax=None,
                        patch_kwargs={'facecolor': 'white'},
                        line_kwargs={'color': 'black'},

                        ):
        if ax is None:
            ax = plt.gca()

        # x-coordinates in axis coordinates, y coordinates in data coordinates
        trans = transforms.blended_transform_factory(
            ax.transAxes, ax.transData)

        # a bar indicting the range of values
        rect = Rectangle((offset, ymin), width=width, height=ymax -
                         ymin, transform=trans, clip_on=False, **patch_kwargs)
        ax.add_patch(rect)

        # delimiters at the start and end of the range mimicking ticks
        min_delimiter = Line2D((offset+width, offset), (ymin, ymin),
                               transform=trans, clip_on=False, **line_kwargs, linewidth=.7)
        max_delimiter = Line2D((offset+width, offset), (ymax, ymax),
                               transform=trans, clip_on=False, **line_kwargs, linewidth=.7)
        mid_delimiter = Line2D((offset+width, offset + width), (ymin, ymax),
                               transform=trans, clip_on=False, **line_kwargs, linewidth=.7)
        ax.add_artist(min_delimiter)
        ax.add_artist(max_delimiter)
        ax.add_artist(mid_delimiter)

        # label
        if label:
            x = offset + 0.5 * width
            y = ymin + 0.5 * (ymax - ymin)
            # we need to fix the alignment as otherwise our choice of x and y leads to unexpected results;
            # e.g. 'right' does not align with the minimum_delimiter
            ax.text(x, y, label, fontsize=fontsize, horizontalalignment='center', verticalalignment='center',
                    clip_on=False, transform=trans, **text_kwargs)  # Y-LABELS Valores and Rotation
            # Para cambiar el tamaño de letra de las XLABELS => fontsize=15
            # Para poner negrita =>  fontweight='bold'
            # Para poner cursiva =>  style='italic'
            # Para darle color al texto => color='green' o color='valor-hexadecimal'

    def addYticks(self):
        temp_labels = []
        temp_labelsPos = []
        labelPos = []
        cnt = 1
        for idx in range(len(self.labels)):
            if idx == 0:
                temp_labels.append(self.labels[idx])
                continue
            if self.labels[idx] == self.labels[idx-1]:
                cnt += 1
            else:
                temp_labels.append(self.labels[idx])
                temp_labelsPos.append(cnt)
                cnt = 1
        temp_labelsPos.append(cnt)
        start_pos = self.nPackages + .5
        for pos in temp_labelsPos:
            if start_pos == self.nPackages + .5:
                labelPos.append(
                    (start_pos + .5, start_pos - self.nPackages * pos / len(self.labels) + .15))
                start_pos = start_pos - self.nPackages * \
                    pos / len(self.labels) + .15
            else:
                if start_pos - self.nPackages * pos / len(self.labels) < 0.7:
                    labelPos.append(
                        (start_pos, 0.5))
                else:
                    labelPos.append(
                        (start_pos, start_pos - self.nPackages * pos / len(self.labels)))
                start_pos = start_pos - self.nPackages * \
                    pos / len(self.labels)
        temp_mlabels = []
        temp_mlabelsPos = []
        mlabelPos = []
        cnt = 1
        for idx in range(len(self.main_Lables)):
            if idx == 0:
                temp_mlabels.append(self.main_Lables[idx])
                continue
            if self.main_Lables[idx] == self.main_Lables[idx-1]:
                cnt += 1
            else:
                temp_mlabels.append(self.main_Lables[idx])
                temp_mlabelsPos.append(cnt)
                cnt = 1
        temp_mlabelsPos.append(cnt)
        start_pos = self.nPackages + .5
        for pos in temp_mlabelsPos:
            if start_pos == self.nPackages + .5:
                mlabelPos.append(
                    (start_pos + .5, start_pos - self.nPackages * pos / len(self.main_Lables)))
                start_pos = start_pos - self.nPackages * \
                    pos / len(self.main_Lables)
            else:
                if start_pos - self.nPackages * pos / len(self.main_Lables) < 0.7:
                    mlabelPos.append(
                        (start_pos, .5))
                else:
                    mlabelPos.append(
                        (start_pos, start_pos - self.nPackages * pos / len(self.main_Lables)))
                start_pos = start_pos - self.nPackages * \
                    pos / len(self.main_Lables)
        width = -0.12
        offsets = [0, -0.12]
        for ii, (level, offset) in enumerate(zip((labelPos, mlabelPos), offsets)):
            for jj, (ymin, ymax) in enumerate(level):
                if ii == 0:
                    temp_labels[jj] = '\n'.join(temp_labels[jj][i:i+17]
                                                for i in range(0, len(temp_labels[jj]), 17))  # label split by 17 characters
                    self.annotate_yrange(
                        ymin, ymax,  temp_labels[jj], fontsize=10, offset=offset, width=width, text_kwargs={'rotation': 0})  # fontsize de Subproject
                else:
                    temp_mlabels[jj] = '\n'.join(temp_mlabels[jj][i:i+17]
                                                 for i in range(0, len(temp_mlabels[jj]), 17))  # label split by 50 characters
                    self.annotate_yrange(
                        ymin, ymax,  temp_mlabels[jj], fontsize=10, offset=offset, width=width, text_kwargs={'rotation': 0})  # fontsize de Project

    def add_fill_betweenx(self):
        start_d = datetime.strptime(self.btx_start, "%d/%m/%Y")
        end_d = datetime.strptime(self.btx_end, "%d/%m/%Y")
        ytic = self.yPos
        ytic = np.insert(ytic, 0, self.nPackages + 1)
        ytic = np.append(ytic, 0)
        plt.fill_betweenx(ytic, self.xticks[self.xticks.index(datetime.date(datetime(start_d.year, start_d.month, 1)))],
                          self.xticks[self.xticks.index(datetime.date(datetime(end_d.year, end_d.month, 1)))], color='lightblue', alpha=.5)

    def render(self):
        # init figure
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0.2)
        self.ax.yaxis.grid(False)
        self.ax.xaxis.grid(False)
        self.fig.set_size_inches(self.size)
        # assemble colors
        colors = []
        # edgecolor = []

        # bl_edgecolor = []

        edges = []
        bl_edges = []
        ed_left = []
        ed_durations = []
        bl_ed_durations = []
        bl_ed_left = []
        idx = 0
        bl_durations = list(self.bldurations)
        durations = list(self.durations)
        for pkg in self.packages:
            colors.append(pkg.color)
            if len(pkg.milestones) == 0 and len(pkg.bl_milestones) == 0:
                edges.append(self.yPos[idx])
                ed_durations.append(durations[idx])
                ed_left.append(self.start[idx])
                bl_edges.append(self.yPos[idx])
                bl_ed_durations.append(bl_durations[idx])
                bl_ed_left.append(self.blstart[idx])
            # else:
            #     edgecolor.append("black")
            idx += 1
            # else:
            #     bl_edgecolor.append("black")
            # add tasks of delay ="YES"
        self.barlist = plt.barh(bl_edges, bl_ed_durations, left=bl_ed_left, align='center',
                                height=.4, alpha=1, color='#A9A9A9', edgecolor="black")  # BASELINE TASK
        # self.barlist = plt.barh(self.yPos, list(
        #     self.bldurations), left=self.blstart, align='center', height=.4, alpha=1, color='#A9A9A9', edgecolor="black")  # BASELINE TASK Shape, fill, border, color
        # add main tasks to figure
        self.barlist = plt.barh(edges, ed_durations, left=ed_left, align='center',
                                height=.4, alpha=1, color=colors, edgecolor="black")  # TASK Shape, fill,
        # self.barlist = plt.barh(self.yPos, list(
        #     self.durations), left=self.start, align='center', height=.4, alpha=1, color=colors, edgecolor="black")  # TASK Shape, fill, border, color. El color del fill es `colors` esta parametrizado y el dato sale de las funciones secundarias

        # format plot
        self.format()
        self.addYticks()
        self.add_milestones()
        self.add_blmilestones()
        self.add_fill_betweenx()

        self.add_dashline()
        self.add_legend()
        self.save(self.pdf, self.title)

    @staticmethod
    def show():
        i = 1  # Pongo i = 1 porque no podemos dejar el metodo vacío
        # plt.show()

    @staticmethod
    def save(pdf, title):
        filename = 'output/'+str(title)
        if pdf == False:
            plt.savefig(filename+'.png', bbox_inches='tight', dpi=1000)
        else:
            plt.savefig(filename+'.pdf', bbox_inches='tight')
