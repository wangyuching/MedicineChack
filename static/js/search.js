let allData = [];
let groupedDataArray = [];
let currentPage = 1;

let currentCalYear = new Date().getFullYear();
let currentCalMonth = new Date().getMonth();

function refreshTable() {
    fetch('/api/history')
        .then(response => response.json())
        .then(data => {
            allData = [...data].reverse();
            groupAllDataByDate();
            renderPage();
            renderCalendar();
        })
        .catch(error => console.error('Error fetching data:', error));
}

function groupAllDataByDate() {
    const groups = [];

    allData.forEach((row, index) => {
        const isLatest = index === 0;
        const dataStr = row.dt.split(" ")[0] || "未知日期";
        const timeStr = row.dt.split(" ")[1] || "未知時間";

        let existingGroup = groups.find(group => group.data === dataStr);
        if (!existingGroup) {
            existingGroup = { data: dataStr, records: [] };
            groups.push(existingGroup);
        }
        existingGroup.records.push({ ...row, timeStr, isLatest });
    });

    groupedDataArray = groups;
}

function renderPage() {
    const logContent = document.getElementById("log-content");

    if (groupedDataArray.length === 0) {
        logContent.innerHTML = '<div class="text-center fs-5">目前沒有記錄。</div>';
        const pageInfo = document.getElementById("page-info");
        if (pageInfo) pageInfo.innerHTML = '<span class="fw-bold text-dark mx-2">第 0 頁</span>';
        document.getElementById("btn-prev").disabled = true;
        document.getElementById("btn-next").disabled = true;
        return;
    }

    const totalPages = groupedDataArray.length;
    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const currentGroup = groupedDataArray[currentPage - 1];

    if (currentGroup && currentGroup.data !== "未知日期") {
        const dateParts = currentGroup.data.split("-");
        if (dateParts.length === 3) {
        }
    }

    let htmlContent = "";
    htmlContent += `
            <div class="data-group">
                <div class="data-title">
                    <span>${currentGroup.data}</span>
                    <span class="badge bg-secondary fs-6">共 ${currentGroup.records.length} 筆資料</span>
                </div>
                <div class="record-grid">
        `;

    currentGroup.records.forEach(row => {
        const badge = row.isLatest ? '<span class="badge bg-success ms-1">最新紀錄</span>' : '';

        htmlContent += `
                <div class="record-card">
                    <div class="record-time">${row.timeStr}${badge}</div>
                    <div class="img-container">
            `;

        if (row.img) {
            htmlContent += `<img src="data:image/jpeg;base64,${row.img}" alt="Record Image">`;
        } else {
            htmlContent += `<div class="no-image">沒有圖片</div>`;
        }

        htmlContent += `
                    </div>
                </div>
            `;
    });

    htmlContent += `
                </div>
            </div>
        `;

    logContent.innerHTML = htmlContent;

    renderPaginationButtons(totalPages);

    renderCalendar();
}

function renderPaginationButtons(totalPages) {
    const pageInfo = document.getElementById("page-info");
    let paginationHtml = "";

    if (totalPages <= 5) {
        for (let i = 1; i <= totalPages; i++) {
            paginationHtml += createPageButton(i, i === currentPage);
        }
    } else {
        paginationHtml += createPageButton(1, 1 === currentPage);

        if (currentPage > 3) {
            paginationHtml += `<span class="text-muted px-1">...</span>`;
        }

        let startPage = Math.max(2, currentPage - 1);
        let endPage = Math.min(totalPages - 1, currentPage + 1);

        if (currentPage <= 3) {
            endPage = 4;
        } else if (currentPage >= totalPages - 2) {
            startPage = totalPages - 3;
        }

        for (let i = startPage; i <= endPage; i++) {
            paginationHtml += createPageButton(i, i === currentPage);
        }

        if (currentPage < totalPages - 2) {
            paginationHtml += `<span class="text-muted px-1">...</span>`;
        }

        paginationHtml += createPageButton(totalPages, totalPages === currentPage);
    }

    pageInfo.innerHTML = paginationHtml;

    document.getElementById("btn-prev").disabled = currentPage === 1;
    document.getElementById("btn-next").disabled = currentPage === totalPages || totalPages === 0;
}

function createPageButton(pageNum, isActive) {
    const btnClass = isActive ? 'btn btn-primary page-num-btn' : 'btn btn-outline-primary page-num-btn';
    return `<button class="${btnClass}" onclick="goToPage(${pageNum})">${pageNum}</button>`;
}

function goToPage(pageNum) {
    currentPage = pageNum;
    renderPage();
}

function changePagination(offset) {
    currentPage += offset;
    renderPage();
}

function toggleCalendar() {
    const cal = document.getElementById("floating-calendar");
    if (cal.style.display === "block") {
        cal.style.display = "none";
    } else {
        if (groupedDataArray.length > 0 && groupedDataArray[currentPage - 1]) {
            const currentDateStr = groupedDataArray[currentPage - 1].data;
            const parts = currentDateStr.split("-");
            if (parts.length === 3) {
                currentCalYear = parseInt(parts[0]);
                currentCalMonth = parseInt(parts[1]) - 1;
            }
        }
        cal.style.display = "block";
        renderCalendar();
    }
}

function moveMonth(direction) {
    currentCalMonth += direction;
    if (currentCalMonth > 11) {
        currentCalMonth = 0;
        currentCalYear++;
    } else if (currentCalMonth < 0) {
        currentCalMonth = 11;
        currentCalYear--;
    }
    renderCalendar();
}

function renderCalendar() {
    const grid = document.getElementById("calendar-grid");
    const monthYearLabel = document.getElementById("calendar-month-year");

    if (!grid) return;

    monthYearLabel.textContent = `${currentCalYear} 年 ${String(currentCalMonth + 1).padStart(2, '0')} 月`;

    grid.innerHTML = "";

    const weekLabels = ["日", "一", "二", "三", "四", "五", "六"];
    weekLabels.forEach(label => {
        grid.innerHTML += `<div class="calendar-day-label">${label}</div>`;
    });

    const firstDayIndex = new Date(currentCalYear, currentCalMonth, 1).getDay();
    const totalDays = new Date(currentCalYear, currentCalMonth + 1, 0).getDate();

    for (let i = 0; i < firstDayIndex; i++) {
        grid.innerHTML += `<div class="calendar-day"></div>`;
    }

    const currentSelectedDateStr = groupedDataArray[currentPage - 1] ? groupedDataArray[currentPage - 1].data : "";

    for (let day = 1; day <= totalDays; day++) {
        const dateStr = `${currentCalYear}-${String(currentCalMonth + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

        const groupIndex = groupedDataArray.findIndex(group => group.data === dateStr);
        const hasRecord = groupIndex !== -1;
        const isCurrentlySelected = dateStr === currentSelectedDateStr;

        let dayClass = "calendar-day";
        let onClickAttribute = "";

        if (hasRecord) {
            dayClass += " has-record";
            onClickAttribute = `onclick="goToPage(${groupIndex + 1})"`;
        }
        if (isCurrentlySelected) {
            dayClass += " current-selected";
        }

        grid.innerHTML += `<div class="${dayClass}" ${onClickAttribute}>${day}</div>`;
    }
}

document.addEventListener("click", function (event) {
    const cal = document.getElementById("floating-calendar");
    const toggleBtn = document.querySelector(".calendar-toggle-btn");
    if (cal && cal.style.display === "block" && !cal.contains(event.target) && !toggleBtn.contains(event.target)) {
        cal.style.display = "none";
    }
});

setInterval(refreshTable, 5000);

window.onload = refreshTable;
